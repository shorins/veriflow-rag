from __future__ import annotations

import json
import re

from veriflow_rag.core.config import AppConfig, get_config
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.verification.llm import StructuredLLMRunner
from veriflow_rag.verification.models import Claim, RawClaimList
from veriflow_rag.verification.prompting import load_prompt_artifacts, render_user_prompt


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def find_span_range(text: str, span: str) -> tuple[int, int] | None:
    if not span:
        return None
    start = text.find(span)
    if start != -1:
        return start, start + len(span)

    collapsed_text = re.sub(r"\s+", " ", text)
    collapsed_span = re.sub(r"\s+", " ", span).strip()
    start = collapsed_text.find(collapsed_span)
    if start == -1:
        return None

    prefix = collapsed_text[:start]
    true_start = len(prefix.replace(" ", ""))
    stripped = re.sub(r"\s+", "", text)
    stripped_span = re.sub(r"\s+", "", span)
    stripped_start = stripped.find(stripped_span)
    if stripped_start == -1:
        return None

    count = 0
    real_start = 0
    for idx, char in enumerate(text):
        if not char.isspace():
            if count == stripped_start:
                real_start = idx
                break
            count += 1
    real_end = real_start
    remaining = len(stripped_span)
    seen = 0
    for idx in range(real_start, len(text)):
        if not text[idx].isspace():
            seen += 1
        if seen == remaining:
            real_end = idx + 1
            break
    return real_start, real_end


class ClaimExtractor:
    def __init__(self, config: AppConfig, client: LMStudioChatClient | None = None) -> None:
        self.config = config
        self.llm = StructuredLLMRunner(config, client=client)
        self.prompt_artifacts = load_prompt_artifacts("claim_decomposition")

    def extract_claims(self, draft_answer: str) -> list[Claim]:
        sentences = split_sentences(draft_answer)
        sentences_block = "\n".join(
            f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1)
        )
        schema_json = json.dumps(self.prompt_artifacts.output_schema, ensure_ascii=False, indent=2)
        user_prompt = render_user_prompt(
            self.prompt_artifacts.user_template,
            draft_answer=draft_answer,
            sentence_list=sentences_block,
            output_schema=schema_json,
        )
        raw = self.llm.run(
            model_name=getattr(self.config, "verification_model_name", self.config.synthesis_model_name),
            system_prompt=self.prompt_artifacts.system_prompt,
            user_prompt=user_prompt,
            output_schema=self.prompt_artifacts.output_schema,
            temperature=self.config.verification_temperature,
            max_tokens=self.config.verification_max_tokens,
            timeout_seconds=self.config.verification_timeout_seconds,
            response_model=RawClaimList,
        )

        validated: list[Claim] = []
        for index, claim in enumerate(raw.claims, start=1):
            sentence_index = max(0, claim.source_sentence_index - 1)
            fallback_span = sentences[sentence_index] if 0 <= sentence_index < len(sentences) else claim.claim_text
            source_span = claim.source_span if find_span_range(draft_answer, claim.source_span) else fallback_span
            validated.append(
                Claim(
                    claim_id=claim.claim_id or f"c{index}",
                    claim_text=claim.claim_text.strip(),
                    source_span=source_span.strip(),
                    source_sentence_index=claim.source_sentence_index,
                )
            )

        return validated


def build_claim_extractor() -> ClaimExtractor:
    return ClaimExtractor(get_config())
