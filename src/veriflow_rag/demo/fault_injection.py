from __future__ import annotations

import re

from veriflow_rag.core.config import DemoFaultMode
from veriflow_rag.demo.models import FaultInjectionResult, InjectedFaultSpan


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _replace_first(text: str, old: str, new: str) -> str | None:
    if old not in text:
        return None
    return text.replace(old, new, 1)


def _inject_sentence_fault(sentence: str) -> tuple[str, str] | None:
    lowered = sentence.lower()

    if "вспомогатель" in lowered and "процесс" in lowered:
        rewritten = _replace_first(
            sentence,
            "вспомогательным процессом",
            "обязательным основным этапом",
        ) or _replace_first(
            sentence,
            "вспомогательный процесс",
            "обязательный основной этап",
        )
        if rewritten and rewritten != sentence:
            return "category_mix", rewritten

    if ("включает" in lowered or "охватывает" in lowered) and "аудит" not in lowered:
        suffix = " Также к обязательным этапам относится аудит системы."
        if sentence.endswith("."):
            return "unsupported_detail", sentence[:-1] + "." + suffix
        return "unsupported_detail", sentence + suffix

    if "различ" in lowered or "отлич" in lowered:
        return (
            "unsupported_detail",
            sentence + " При этом жизненный цикл программного обеспечения не охватывает эксплуатацию системы.",
        )

    if "," in sentence and any(marker in lowered for marker in ("этап", "процесс", "включает", "состоит")):
        match = re.search(r",\s*([^,]+)\.$", sentence)
        if match:
            last_item = match.group(1).strip()
            if "сертификац" not in lowered:
                rewritten = sentence[: match.start(1)] + "обязательную сертификацию."
                if rewritten != sentence and last_item:
                    return "list_swap", rewritten

    if "может включ" in lowered:
        rewritten = _replace_first(sentence, "может включать", "включает")
        if rewritten and rewritten != sentence:
            return "overgeneralization", rewritten

    return None


def inject_demo_faults(
    *,
    answer: str,
    mode: DemoFaultMode,
    count: int,
) -> FaultInjectionResult:
    if mode == "off" or not answer.strip():
        return FaultInjectionResult()

    sentences = _split_sentences(answer)
    if not sentences:
        return FaultInjectionResult()

    updated_answer = answer
    injected_spans: list[InjectedFaultSpan] = []

    for index, sentence in enumerate(sentences, start=1):
        if len(injected_spans) >= max(1, min(2, count)):
            break
        candidate = _inject_sentence_fault(sentence)
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

    effective_mode: DemoFaultMode = "deterministic" if mode == "model_guided" else mode
    count_value = len(injected_spans)
    return FaultInjectionResult(
        active=True,
        mode=effective_mode,
        count=count_value,
        summary=(
            f"Controlled mismatch demo active: draft contains {count_value} intentionally perturbed "
            f"claim{'s' if count_value > 1 else ''} for verification visualization."
        ),
        answer=updated_answer,
        spans=injected_spans,
    )
