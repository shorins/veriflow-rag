from __future__ import annotations

import json
from dataclasses import dataclass

from veriflow_rag.core.config import AppConfig, VerificationSensitivity, get_config
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.synthesis.models import PreparedEvidence
from veriflow_rag.verification.claims import find_span_range
from veriflow_rag.verification.llm import StructuredLLMRunner
from veriflow_rag.verification.models import (
    AppliedRewrite,
    ClaimVerificationResult,
    RawClaimRewriteResult,
)
from veriflow_rag.verification.prompting import load_prompt_artifacts, render_user_prompt


@dataclass
class RewriteDecision:
    rewrite_triggered: bool
    partial_ratio: float
    problem_span_ratio: float


@dataclass(frozen=True)
class VerificationProfile:
    sensitivity: VerificationSensitivity
    partial_ratio_threshold: float
    problem_span_ratio_threshold: float
    treat_partial_as_rewrite_candidate: bool
    sensitivity_note: str


def select_verification_profile(config: AppConfig) -> VerificationProfile:
    sensitivity = getattr(config, "verification_sensitivity", "balanced")
    base_partial = config.verification_partial_ratio_threshold
    base_span = config.verification_problem_span_ratio_threshold
    if sensitivity == "conservative":
        return VerificationProfile(
            sensitivity=sensitivity,
            partial_ratio_threshold=max(base_partial, 0.35),
            problem_span_ratio_threshold=max(base_span, 0.25),
            treat_partial_as_rewrite_candidate=False,
            sensitivity_note=(
                "Use conservative verification. Mark partial only when important support is missing, "
                "and avoid rewrite unless the claim is materially inaccurate."
            ),
        )
    if sensitivity == "demo":
        return VerificationProfile(
            sensitivity=sensitivity,
            partial_ratio_threshold=min(base_partial, 0.1),
            problem_span_ratio_threshold=min(base_span, 0.08),
            treat_partial_as_rewrite_candidate=True,
            sensitivity_note=(
                "Use demo sensitivity. For comparisons, enumerations, or broad category summaries, "
                "prefer `partial` over `supported` when the evidence is narrower than the claim. "
                "Mark materially broader partial claims as rewrite_needed."
            ),
        )
    return VerificationProfile(
        sensitivity=sensitivity,
        partial_ratio_threshold=base_partial,
        problem_span_ratio_threshold=base_span,
        treat_partial_as_rewrite_candidate=False,
        sensitivity_note=(
            "Use balanced verification. Distinguish supported from partial normally, and request rewrite "
            "only when the claim is materially inaccurate."
        ),
    )


def should_trigger_rewrite(
    draft_answer: str,
    claim_results: list[ClaimVerificationResult],
    partial_ratio_threshold: float,
    problem_span_ratio_threshold: float,
) -> RewriteDecision:
    if not claim_results or not draft_answer.strip():
        return RewriteDecision(False, 0.0, 0.0)

    if any(result.status == "contradicted" for result in claim_results):
        return RewriteDecision(True, 1.0, 1.0)
    if any(result.status == "unsupported" for result in claim_results):
        return RewriteDecision(True, 1.0, 1.0)

    problem_results = [result for result in claim_results if result.status in {"partial", "unsupported", "contradicted"}]
    partial_ratio = len(problem_results) / max(1, len(claim_results))

    problem_chars = 0
    for result in problem_results:
        if result.source_span:
            problem_chars += len(result.source_span)
    problem_span_ratio = problem_chars / max(1, len(draft_answer))

    triggered = (
        partial_ratio >= partial_ratio_threshold
        or problem_span_ratio >= problem_span_ratio_threshold
    )
    return RewriteDecision(triggered, partial_ratio, problem_span_ratio)


class ClaimRewriter:
    def __init__(self, config: AppConfig, client: LMStudioChatClient | None = None) -> None:
        self.config = config
        self.llm = StructuredLLMRunner(config, client=client)
        self.prompt_artifacts = load_prompt_artifacts("claim_rewrite")

    def rewrite_claim(
        self,
        *,
        draft_answer: str,
        claim_result: ClaimVerificationResult,
        evidence_blocks: list[PreparedEvidence],
    ) -> str:
        evidence_xml = "\n\n".join(item.prompt_text for item in evidence_blocks)
        schema_json = json.dumps(self.prompt_artifacts.output_schema, ensure_ascii=False, indent=2)
        user_prompt = render_user_prompt(
            self.prompt_artifacts.user_template,
            draft_answer=draft_answer,
            claim_id=claim_result.claim_id,
            claim_text=claim_result.claim_text,
            source_span=claim_result.source_span,
            reason=claim_result.reason,
            proposed_claim=claim_result.revised_claim or "",
            evidence=evidence_xml,
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
            response_model=RawClaimRewriteResult,
        )
        return raw.rewritten_span.strip()


def apply_rewrites(
    draft_answer: str,
    claim_results: list[ClaimVerificationResult],
    rewritten_spans: dict[str, str],
) -> tuple[str, list[AppliedRewrite]]:
    priority = {"contradicted": 0, "unsupported": 1, "partial": 2}
    candidates = []
    for result in claim_results:
        if not rewritten_spans.get(result.claim_id):
            continue
        span_range = find_span_range(draft_answer, result.source_span)
        if span_range is None:
            continue
        candidates.append((priority.get(result.status, 99), span_range[0], span_range[1], result))

    candidates.sort(key=lambda item: (item[0], item[1]))

    accepted: list[tuple[int, int, ClaimVerificationResult]] = []
    for _, start, end, result in candidates:
        overlaps = any(not (end <= existing_start or start >= existing_end) for existing_start, existing_end, _ in accepted)
        if overlaps:
            continue
        accepted.append((start, end, result))

    accepted.sort(key=lambda item: item[0])
    pieces: list[str] = []
    rewrites: list[AppliedRewrite] = []
    cursor = 0
    for start, end, result in accepted:
        pieces.append(draft_answer[cursor:start])
        new_span = rewritten_spans[result.claim_id]
        pieces.append(new_span)
        rewrites.append(
            AppliedRewrite(
                claim_id=result.claim_id,
                old_span=draft_answer[start:end],
                new_span=new_span,
                status_before=result.status,
            )
        )
        cursor = end
    pieces.append(draft_answer[cursor:])
    return "".join(pieces), rewrites


def build_claim_rewriter() -> ClaimRewriter:
    return ClaimRewriter(get_config())
