from __future__ import annotations

import json
from dataclasses import dataclass
import re

from pydantic import ValidationError

from veriflow_rag.core.config import AppConfig, DraftStrategy, get_config
from veriflow_rag.retrieval.pipeline import EvidenceBlock, build_retriever
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.synthesis.models import (
    AnswerDepth,
    Citation,
    PreparedEvidence,
    RawSynthesizedAnswer,
    SynthesizedAnswer,
    SynthesisResultBundle,
)
from veriflow_rag.synthesis.prompts import load_prompt_artifacts, render_user_prompt
from veriflow_rag.tools.json_extract import extract_json_payload


@dataclass(frozen=True)
class SynthesisProfile:
    answer_depth: AnswerDepth
    top_evidence_k: int
    max_evidence_chars: int
    draft_strategy: DraftStrategy
    strategy_note: str
    content_plan: str


class AnswerSynthesisService:
    def __init__(
        self,
        config: AppConfig,
        client: LMStudioChatClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or LMStudioChatClient(config)
        self.prompt_artifacts = load_prompt_artifacts()

    @staticmethod
    def _select_evidence(blocks: list[EvidenceBlock], top_k: int) -> list[EvidenceBlock]:
        preferred = [block for block in blocks if block.confidence_label in {"high", "medium"}]
        fallback = [block for block in blocks if block.confidence_label not in {"high", "medium"}]
        ordered = preferred + fallback
        return ordered[:top_k]

    @staticmethod
    def _classify_query_intent(query: str) -> str:
        normalized = query.lower()
        if any(marker in normalized for marker in ("что такое", "что представляет собой", "определение", "what is", "define")):
            return "definition"
        if any(marker in normalized for marker in ("какие", "перечис", "список", "виды", "типы", "элементы")):
            return "enumeration"
        if any(marker in normalized for marker in ("разница", "различ", "отлич", "сравн", "difference", "vs", "versus")):
            return "comparison"
        if any(marker in normalized for marker in ("как", "порядок", "этап", "шаг", "процед", "алгоритм", "workflow")):
            return "procedure"
        return "factoid"

    @staticmethod
    def classify_answer_depth(query: str) -> AnswerDepth:
        normalized = " ".join(query.lower().split())
        detailed_patterns = (
            "подробно",
            "расскажи",
            "расскажи про",
            "опиши",
            "раскрой",
            "объясни",
            "что входит",
            "из чего состоит",
            "какие этапы",
            "какие стадии",
            "как устроен",
            "как устроена",
            "как устроено",
            "как происходит",
            "как работает",
            "что включает",
        )
        brief_patterns = (
            "что такое",
            "кто такой",
            "что определяет",
            "что означает",
            "что представляет собой",
            "какая",
            "какой",
            "какое",
            "когда",
            "где",
            "сколько",
        )
        if any(pattern in normalized for pattern in detailed_patterns):
            return "detailed"
        if any(pattern in normalized for pattern in brief_patterns):
            return "brief"
        return "standard"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]

    @staticmethod
    def _format_paragraphs(answer: str, answer_depth: AnswerDepth, draft_strategy: DraftStrategy) -> str:
        if not answer.strip():
            return answer
        if answer_depth == "brief":
            return re.sub(r"\s+", " ", answer).strip()

        if "\n\n" in answer and answer_depth == "detailed":
            return answer.strip()

        sentences = AnswerSynthesisService._split_sentences(answer)
        if len(sentences) < 2:
            return answer.strip()

        if answer_depth == "standard":
            midpoint = max(1, len(sentences) // 2)
            groups = [sentences[:midpoint], sentences[midpoint:]]
        else:
            if len(sentences) >= 6:
                size = max(2, (len(sentences) + 2) // 3)
                groups = [sentences[i:i + size] for i in range(0, len(sentences), size)]
            elif len(sentences) >= 3:
                groups = [[sentence] for sentence in sentences]
            else:
                groups = [sentences[:1], sentences[1:]]

        paragraphs = [" ".join(group).strip() for group in groups if group]
        if draft_strategy == "demo" and answer_depth in {"standard", "detailed"} and len(paragraphs) == 1:
            return answer.strip()
        return "\n\n".join(paragraphs).strip()

    def _strategy_note(self, answer_depth: AnswerDepth, strategy: DraftStrategy, query_intent: str) -> str:
        if strategy == "conservative":
            return "Prefer precision over coverage. Avoid broad summaries unless every part is directly supported."
        if strategy == "demo":
            if query_intent in {"comparison", "procedure", "enumeration"} or answer_depth == "detailed":
                return (
                    "Produce a structured overview with explicit sentence-level claims. "
                    "Use 3 short paragraphs when possible, keep one substantive claim per sentence, and prefer slightly broader "
                    "but still evidence-grounded phrasing that can be independently verified."
                )
            return (
                "Use a richer, conference-demo friendly answer shape with multiple grounded sentences "
                "instead of a single compressed summary."
            )
        return "Balance precision and coverage. Prefer short but complete grounded explanations."

    def _content_plan(self, query: str, answer_depth: AnswerDepth, query_intent: str) -> str:
        normalized = query.lower()
        if answer_depth == "brief":
            return "Give one short paragraph with the main grounded answer only."
        if query_intent == "comparison" or any(marker in normalized for marker in ("различ", "отлич", "сравн")):
            return (
                "Paragraph 1: briefly define or frame both concepts. "
                "Paragraph 2: explain the key differences using only supported distinctions. "
                "Paragraph 3: add any important overlap, scope note, or limitation that is explicitly supported."
            )
        if any(marker in normalized for marker in ("этап", "стад", "жизненн", "процесс", "входит", "включает")):
            return (
                "Paragraph 1: define the lifecycle or process and its scope. "
                "Paragraph 2: describe the main stages or core processes in grounded detail. "
                "Paragraph 3: explain supporting or auxiliary processes, and explicitly note if the evidence does not fully classify them as mandatory or auxiliary."
            )
        return (
            "Paragraph 1: brief framing or definition. "
            "Paragraph 2: main elements, stages, or components. "
            "Paragraph 3: clarifications, supporting processes, or important constraints that are explicitly supported."
        )

    def select_synthesis_profile(self, query: str, answer_depth: AnswerDepth) -> SynthesisProfile:
        query_intent = self._classify_query_intent(query)
        strategy = getattr(self.config, "draft_strategy", "balanced")
        effective_depth = answer_depth
        if strategy == "demo" and answer_depth == "standard" and query_intent in {"comparison", "procedure", "enumeration"}:
            effective_depth = "detailed"

        brief_top_k = self.config.synthesis_top_evidence_k
        brief_chars = self.config.synthesis_max_evidence_chars
        if strategy == "conservative":
            brief_top_k = max(3, brief_top_k - 1)
            brief_chars = max(900, int(brief_chars * 0.9))
        elif strategy == "demo":
            brief_top_k += 1
            brief_chars = int(brief_chars * 1.1)

        if effective_depth == "brief":
            return SynthesisProfile(
                answer_depth="brief",
                top_evidence_k=brief_top_k,
                max_evidence_chars=brief_chars,
                draft_strategy=strategy,
                strategy_note=self._strategy_note(effective_depth, strategy, query_intent),
                content_plan=self._content_plan(query, effective_depth, query_intent),
            )
        if effective_depth == "standard":
            return SynthesisProfile(
                answer_depth="standard",
                top_evidence_k=brief_top_k + 2,
                max_evidence_chars=max(int(brief_chars * 1.35), 1600),
                draft_strategy=strategy,
                strategy_note=self._strategy_note(effective_depth, strategy, query_intent),
                content_plan=self._content_plan(query, effective_depth, query_intent),
            )
        return SynthesisProfile(
            answer_depth="detailed",
            top_evidence_k=max(brief_top_k + 4, 8),
            max_evidence_chars=max(int(brief_chars * 1.9), 2400),
            draft_strategy=strategy,
            strategy_note=self._strategy_note(effective_depth, strategy, query_intent),
            content_plan=self._content_plan(query, effective_depth, query_intent),
        )

    def _prepare_evidence(self, blocks: list[EvidenceBlock], profile: SynthesisProfile) -> list[PreparedEvidence]:
        selected = self._select_evidence(blocks, profile.top_evidence_k)
        prepared: list[PreparedEvidence] = []
        for index, block in enumerate(selected, start=1):
            text = block.expanded_text.strip()
            if len(text) > profile.max_evidence_chars:
                text = text[: profile.max_evidence_chars].rstrip() + "..."
            prepared.append(
                PreparedEvidence(
                    evidence_id=f"ev_{index}",
                    block=block,
                    prompt_text=(
                        f"<evidence_item id=\"ev_{index}\">\n"
                        f"<metadata>\n"
                        f"file_name: {block.file_name}\n"
                        f"section_title: {block.section_title}\n"
                        f"page_span: {block.page_span}\n"
                        f"confidence: {block.confidence_label}\n"
                        f"</metadata>\n"
                        f"<content>\n{text}\n</content>\n"
                        f"</evidence_item>"
                    ),
                )
            )
        return prepared

    def _has_sufficient_context(self, query: str, prepared: list[PreparedEvidence]) -> bool:
        confident = sum(
            1 for item in prepared if item.block.confidence_label in {"high", "medium"}
        )
        high_confident = any(item.block.confidence_label == "high" for item in prepared)
        intent = self._classify_query_intent(query)

        if intent in {"definition", "factoid"}:
            return confident >= 1 and high_confident

        return confident >= self.config.synthesis_min_confident_evidence

    def _insufficient_result(
        self,
        *,
        reason: str,
        prepared: list[PreparedEvidence],
        answer_depth: AnswerDepth,
    ) -> SynthesizedAnswer:
        citations = [
            Citation(
                evidence_id=item.evidence_id,
                file_name=item.block.file_name,
                section_title=item.block.section_title,
                support=item.block.expanded_text[:220].strip(),
            )
        for item in prepared[:1]
        ]
        return SynthesizedAnswer(
            answer="Недостаточно подтвержденного контекста, чтобы дать надежный ответ только по найденным фрагментам.",
            citations=citations,
            used_evidence_ids=[item.evidence_id for item in prepared[:1]],
            insufficient_context=True,
            omitted_points=[reason],
            answer_depth=answer_depth,
            model_name=getattr(self.config, "draft_model_name", self.config.synthesis_model_name),
            prompt_version=self.prompt_artifacts.version,
        )

    @staticmethod
    def _validate_ids(raw: RawSynthesizedAnswer, prepared: list[PreparedEvidence]) -> None:
        known_ids = {item.evidence_id for item in prepared}
        unknown_used = set(raw.used_evidence_ids) - known_ids
        if unknown_used:
            raise ValueError(f"Unknown used_evidence_ids: {sorted(unknown_used)}")

        unknown_cited = {citation.evidence_id for citation in raw.citations} - known_ids
        if unknown_cited:
            raise ValueError(f"Unknown citation evidence_ids: {sorted(unknown_cited)}")

    def _normalize_result(
        self,
        raw: RawSynthesizedAnswer,
        prepared: list[PreparedEvidence],
        answer_depth: AnswerDepth,
        draft_strategy: DraftStrategy,
    ) -> SynthesizedAnswer:
        self._validate_ids(raw, prepared)
        evidence_map = {item.evidence_id: item for item in prepared}

        citations: list[Citation] = []
        for citation in raw.citations:
            item = evidence_map[citation.evidence_id]
            citations.append(
                Citation(
                    evidence_id=citation.evidence_id,
                    file_name=item.block.file_name,
                    section_title=item.block.section_title,
                    support=citation.support.strip(),
                )
            )

        return SynthesizedAnswer(
            answer=self._format_paragraphs(raw.answer.strip(), answer_depth, draft_strategy),
            citations=citations,
            used_evidence_ids=raw.used_evidence_ids,
            insufficient_context=raw.insufficient_context,
            omitted_points=raw.omitted_points,
            answer_depth=answer_depth,
            model_name=getattr(self.config, "draft_model_name", self.config.synthesis_model_name),
            prompt_version=self.prompt_artifacts.version,
        )

    def _invoke_model(
        self,
        query: str,
        prepared: list[PreparedEvidence],
        *,
        answer_depth: AnswerDepth,
        profile: SynthesisProfile,
        retry: bool = False,
    ) -> RawSynthesizedAnswer:
        evidence_xml = "\n\n".join(item.prompt_text for item in prepared)
        schema_json = json.dumps(
            self.prompt_artifacts.output_schema,
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = render_user_prompt(
            self.prompt_artifacts.user_template
            + (
                "\n\n<format_reminder>\nReturn only valid JSON that strictly follows the schema.\n</format_reminder>"
                if retry
                else ""
            ),
            query=query,
            answer_depth=answer_depth,
            draft_strategy=profile.draft_strategy,
            strategy_note=profile.strategy_note,
            content_plan=profile.content_plan,
            evidence=evidence_xml,
            schema_json=schema_json,
        )
        raw_text = self.client.chat_json(
            model_name=getattr(self.config, "draft_model_name", self.config.synthesis_model_name),
            system_prompt=self.prompt_artifacts.system_prompt,
            user_prompt=user_prompt,
            output_schema=self.prompt_artifacts.output_schema,
            temperature=self.config.synthesis_temperature,
            max_tokens=self.config.synthesis_max_tokens,
            timeout_seconds=self.config.synthesis_timeout_seconds,
        )
        payload = extract_json_payload(raw_text)
        return RawSynthesizedAnswer.model_validate(payload)

    def synthesize_answer(
        self,
        query: str,
        evidence_blocks: list[EvidenceBlock],
    ) -> SynthesisResultBundle:
        answer_depth = self.classify_answer_depth(query)
        profile = self.select_synthesis_profile(query, answer_depth)
        prepared = self._prepare_evidence(evidence_blocks, profile)
        if not prepared or not self._has_sufficient_context(query, prepared):
            return SynthesisResultBundle(
                query=query,
                evidence_blocks=prepared,
                synthesized_answer=self._insufficient_result(
                    reason="retrieval вернул слишком мало уверенных evidence blocks",
                    prepared=prepared,
                    answer_depth=answer_depth,
                ),
            )

        last_error: Exception | None = None
        for retry in (False, True):
            try:
                raw = self._invoke_model(query, prepared, answer_depth=profile.answer_depth, profile=profile, retry=retry)
                normalized = self._normalize_result(raw, prepared, profile.answer_depth, profile.draft_strategy)
                if not normalized.answer.strip() and not normalized.insufficient_context:
                    raise ValueError("Model returned an empty answer without insufficient_context=true.")
                return SynthesisResultBundle(
                    query=query,
                    evidence_blocks=prepared,
                    synthesized_answer=normalized,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc

        return SynthesisResultBundle(
            query=query,
            evidence_blocks=prepared,
            synthesized_answer=SynthesizedAnswer(
                answer="Недостаточно надежного структурированного вывода от локальной модели, чтобы вернуть честный ответ.",
                citations=[],
                used_evidence_ids=[],
                insufficient_context=True,
                omitted_points=[f"Structured output validation failed: {last_error}"],
                answer_depth=profile.answer_depth,
                model_name=getattr(self.config, "draft_model_name", self.config.synthesis_model_name),
                prompt_version=self.prompt_artifacts.version,
            ),
        )

    def run_query(self, query: str) -> SynthesisResultBundle:
        retriever = build_retriever(use_legacy=False)
        evidence_blocks = retriever.search(query)
        return self.synthesize_answer(query, evidence_blocks)


def build_synthesis_service(config: AppConfig | None = None) -> AnswerSynthesisService:
    return AnswerSynthesisService(config or get_config())
