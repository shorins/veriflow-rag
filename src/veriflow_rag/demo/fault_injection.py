from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from veriflow_rag.core.config import AppConfig, DemoFaultMode
from veriflow_rag.demo.models import (
    FaultInjectionResult,
    InjectedFaultSpan,
    RawFaultInjectionPlan,
)
from veriflow_rag.verification.claims import Claim, find_span_range, split_sentences
from veriflow_rag.verification.llm import StructuredLLMRunner
from veriflow_rag.verification.models import ClaimVerificationResult
from veriflow_rag.verification.prompting import load_prompt_artifacts, render_user_prompt

if TYPE_CHECKING:
    from veriflow_rag.synthesis.client import LMStudioChatClient
    from veriflow_rag.synthesis.models import PreparedEvidence


def _replace_first(text: str, old: str, new: str) -> str | None:
    if old not in text:
        return None
    return text.replace(old, new, 1)


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip().lower()


def _template_inject_sentence_fault(sentence: str) -> tuple[str, str] | None:
    lowered = sentence.lower()

    if "может включ" in lowered:
        rewritten = _replace_first(sentence, "может включать", "включает")
        if rewritten and rewritten != sentence:
            return "overgeneralization", rewritten

    if "помогает" in lowered:
        rewritten = _replace_first(sentence, "помогает", "гарантирует")
        if rewritten and rewritten != sentence:
            return "overgeneralization", rewritten

    if "необходим" in lowered:
        rewritten = sentence.rstrip(".") + ", причём сам по себе он полностью устраняет все связанные риски."
        if rewritten != sentence:
            return "unsupported_detail", rewritten

    if ("включает" in lowered or "охватывает" in lowered or "перечис" in lowered) and "," in sentence:
        match = re.search(r",\s*([^,]+)\.$", sentence)
        if match:
            rewritten = sentence[: match.start(1)] + "полную замену всех процедур контроля."
            if rewritten != sentence:
                return "list_swap", rewritten

    return None


class DemoFaultInjector:
    def __init__(self, config: AppConfig, client: LMStudioChatClient | None = None) -> None:
        self.config = config
        self.llm = StructuredLLMRunner(config, client=client)
        self.prompt_artifacts = load_prompt_artifacts("demo_fault_injection")

    def generate(
        self,
        *,
        query: str,
        answer: str,
        evidence_blocks: list[PreparedEvidence],
        count: int,
    ) -> FaultInjectionResult:
        sentences = split_sentences(answer)
        if not sentences:
            return FaultInjectionResult()

        sentence_list = "\n".join(
            f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1)
        )
        evidence_xml = "\n\n".join(item.prompt_text for item in evidence_blocks)
        user_prompt = render_user_prompt(
            self.prompt_artifacts.user_template,
            query=query,
            draft_answer=answer,
            sentence_list=sentence_list,
            evidence=evidence_xml,
            fault_count=str(max(1, min(2, count))),
            output_schema=json.dumps(self.prompt_artifacts.output_schema, ensure_ascii=False, indent=2),
        )
        raw = self.llm.run(
            model_name=getattr(self.config, "verification_model_name", self.config.synthesis_model_name),
            system_prompt=self.prompt_artifacts.system_prompt,
            user_prompt=user_prompt,
            output_schema=self.prompt_artifacts.output_schema,
            temperature=min(max(self.config.synthesis_temperature, 0.1), 0.3),
            max_tokens=min(max(self.config.synthesis_max_tokens, 500), 900),
            timeout_seconds=self.config.synthesis_timeout_seconds,
            response_model=RawFaultInjectionPlan,
        )
        return _normalize_fault_plan(
            answer=answer,
            faults=raw.faults,
            mode="deterministic",
            requested_count=count,
        )


def _normalize_fault_plan(
    *,
    answer: str,
    faults: list,
    mode: DemoFaultMode,
    requested_count: int,
) -> FaultInjectionResult:
    sentences = split_sentences(answer)
    updated_answer = answer
    injected_spans: list[InjectedFaultSpan] = []
    seen_originals: set[str] = set()

    for index, fault in enumerate(faults, start=1):
        if len(injected_spans) >= max(1, min(2, requested_count)):
            break

        sentence_index = max(1, fault.source_sentence_index)
        fallback_original = (
            sentences[sentence_index - 1]
            if 0 < sentence_index <= len(sentences)
            else fault.original_span
        )
        original_span = (fault.original_span or fallback_original).strip()
        injected_span = fault.injected_span.strip()

        if not original_span or not injected_span or injected_span == original_span:
            continue
        if original_span in seen_originals:
            continue
        if find_span_range(updated_answer, original_span) is None:
            continue
        if _normalize_text(injected_span) == _normalize_text(original_span):
            continue

        updated_answer = updated_answer.replace(original_span, injected_span, 1)
        injected_spans.append(
            InjectedFaultSpan(
                claim_id=f"demo_fault_{index}",
                fault_type=fault.fault_type,
                original_span=original_span,
                injected_span=injected_span,
                source_sentence_index=sentence_index,
            )
        )
        seen_originals.add(original_span)

    if not injected_spans:
        return FaultInjectionResult()

    return FaultInjectionResult(
        active=True,
        mode=mode,
        count=len(injected_spans),
        summary=(
            f"Controlled mismatch demo active: draft contains {len(injected_spans)} intentionally perturbed "
            f"claim{'s' if len(injected_spans) > 1 else ''} for verification visualization."
        ),
        answer=updated_answer,
        spans=injected_spans,
    )


def _fallback_injection(*, answer: str, mode: DemoFaultMode, count: int) -> FaultInjectionResult:
    sentences = split_sentences(answer)
    if not sentences:
        return FaultInjectionResult()

    updated_answer = answer
    injected_spans: list[InjectedFaultSpan] = []
    for index, sentence in enumerate(sentences, start=1):
        if len(injected_spans) >= max(1, min(2, count)):
            break
        candidate = _template_inject_sentence_fault(sentence)
        if not candidate:
            continue
        fault_type, injected = candidate
        if injected == sentence or sentence not in updated_answer:
            continue
        updated_answer = updated_answer.replace(sentence, injected, 1)
        injected_spans.append(
            InjectedFaultSpan(
                claim_id=f"demo_fault_{index}",
                fault_type=fault_type,
                original_span=sentence,
                injected_span=injected,
                source_sentence_index=index,
            )
        )

    if not injected_spans:
        return FaultInjectionResult()

    return FaultInjectionResult(
        active=True,
        mode="deterministic" if mode == "model_guided" else mode,
        count=len(injected_spans),
        summary=(
            f"Controlled mismatch demo active: draft contains {len(injected_spans)} intentionally perturbed "
            f"claim{'s' if len(injected_spans) > 1 else ''} for verification visualization."
        ),
        answer=updated_answer,
        spans=injected_spans,
    )


def inject_demo_faults(
    *,
    answer: str,
    mode: DemoFaultMode,
    count: int,
    query: str = "",
    evidence_blocks: list[PreparedEvidence] | None = None,
    config: AppConfig | None = None,
    client: LMStudioChatClient | None = None,
) -> FaultInjectionResult:
    if mode == "off" or not answer.strip():
        return FaultInjectionResult()

    effective_mode: DemoFaultMode = "deterministic" if mode == "model_guided" else mode
    if config is not None and query.strip() and evidence_blocks:
        try:
            injector = DemoFaultInjector(config, client=client)
            result = injector.generate(
                query=query,
                answer=answer,
                evidence_blocks=evidence_blocks,
                count=count,
            )
            if result.active:
                return result
        except Exception:
            pass

    return _fallback_injection(answer=answer, mode=effective_mode, count=count)


def merge_injected_claims(
    *,
    draft_answer: str,
    claims: list[Claim],
    injected_spans: list[InjectedFaultSpan],
) -> list[Claim]:
    if not injected_spans:
        return claims

    merged = list(claims)
    for injected in injected_spans:
        source_span = (
            injected.injected_span
            if find_span_range(draft_answer, injected.injected_span)
            else injected.original_span
        )
        normalized_span = _normalize_text(source_span)
        already_present = any(
            claim.source_sentence_index == injected.source_sentence_index
            and (
                normalized_span == _normalize_text(claim.source_span)
                or normalized_span in _normalize_text(claim.source_span)
                or _normalize_text(claim.source_span) in normalized_span
            )
            for claim in merged
        )
        if already_present:
            continue
        merged.append(
            Claim(
                claim_id=injected.claim_id,
                claim_text=injected.injected_span.strip(),
                source_span=source_span.strip(),
                source_sentence_index=injected.source_sentence_index,
            )
        )
    return merged


def apply_demo_verification_override(
    result: ClaimVerificationResult,
    injected_spans: list[InjectedFaultSpan],
) -> ClaimVerificationResult:
    if not injected_spans:
        return result

    matching = next(
        (
            injected
            for injected in injected_spans
            if _normalize_text(injected.injected_span) == _normalize_text(result.source_span)
            or _normalize_text(injected.injected_span) == _normalize_text(result.claim_text)
            or _normalize_text(injected.injected_span) in _normalize_text(result.source_span)
            or _normalize_text(injected.injected_span) in _normalize_text(result.claim_text)
            or injected.claim_id == result.claim_id
        ),
        None,
    )
    if matching is None:
        return result

    result.rewrite_needed = True
    if not result.revised_claim:
        result.revised_claim = matching.original_span
    if result.status == "supported":
        result.status = "unsupported"
        result.reason = (
            "В demo-режиме этот фрагмент был намеренно искажен относительно grounded draft; "
            "добавленная формулировка не должна считаться подтверждённой и подлежит переписыванию."
        )
    return result
