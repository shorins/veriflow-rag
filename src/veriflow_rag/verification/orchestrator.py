from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from veriflow_rag.core.config import AppConfig, get_config
from veriflow_rag.demo.fault_injection import apply_demo_verification_override, merge_injected_claims
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.synthesis.models import SynthesisResultBundle
from veriflow_rag.ui.render import render_highlighted_answer
from veriflow_rag.verification.claims import ClaimExtractor
from veriflow_rag.verification.models import (
    ClaimVerificationResult,
    VerificationRunResult,
)
from veriflow_rag.verification.retrieval import ClaimRetrievalService
from veriflow_rag.verification.rewrite import (
    ClaimRewriter,
    apply_rewrites,
    select_rewrite_span,
    select_verification_profile,
    should_trigger_rewrite,
)
from veriflow_rag.verification.verify import ClaimVerifier


ProgressHook = Callable[[str], None]


@dataclass
class VerificationOrchestrator:
    config: AppConfig
    claim_extractor: ClaimExtractor
    retrieval_service: ClaimRetrievalService
    claim_verifier: ClaimVerifier
    claim_rewriter: ClaimRewriter

    def run(
        self,
        query: str,
        draft_bundle: SynthesisResultBundle,
        progress_hook: ProgressHook | None = None,
    ) -> VerificationRunResult:
        def emit(message: str) -> None:
            if progress_hook is not None:
                progress_hook(message)

        draft_answer = draft_bundle.synthesized_answer.answer
        emit("Извлечение claims")
        claims = self.claim_extractor.extract_claims(draft_answer)
        claims = merge_injected_claims(
            draft_answer=draft_answer,
            claims=claims,
            injected_spans=draft_bundle.synthesized_answer.fault_injection_spans,
        )

        claim_results: list[ClaimVerificationResult] = []
        claim_evidence_map: dict[str, list] = {}
        for claim in claims:
            emit(f"Проверка {claim.claim_id}")
            evidence_blocks = self.retrieval_service.retrieve_for_claim(claim.claim_text)
            prepared = self.retrieval_service.prepare_claim_evidence(evidence_blocks)
            claim_evidence_map[claim.claim_id] = prepared
            result = self.claim_verifier.verify_claim(claim, prepared)
            result = apply_demo_verification_override(
                result,
                draft_bundle.synthesized_answer.fault_injection_spans,
            )
            result.rewrite_source_span = select_rewrite_span(draft_answer, result)
            claim_results.append(result)

        verification_profile = select_verification_profile(self.config)
        rewrite_decision = should_trigger_rewrite(
            draft_answer=draft_answer,
            claim_results=claim_results,
            partial_ratio_threshold=verification_profile.partial_ratio_threshold,
            problem_span_ratio_threshold=verification_profile.problem_span_ratio_threshold,
        )

        rewritten_spans: dict[str, str] = {}
        if rewrite_decision.rewrite_triggered:
            emit("Переписывание проблемных фрагментов")
            for result in claim_results:
                if not result.rewrite_needed or result.status == "supported":
                    continue
                prepared = claim_evidence_map.get(result.claim_id, [])
                rewritten = self.claim_rewriter.rewrite_claim(
                    draft_answer=draft_answer,
                    claim_result=result,
                    evidence_blocks=prepared,
                )
                if rewritten:
                    rewritten_spans[result.claim_id] = rewritten

        final_answer, applied_rewrites = apply_rewrites(
            draft_answer=draft_answer,
            claim_results=claim_results,
            rewritten_spans=rewritten_spans,
        )
        highlighted_answer_html = render_highlighted_answer(draft_answer, claim_results)

        return VerificationRunResult(
            draft_answer=draft_answer,
            claims=claims,
            claim_results=claim_results,
            highlighted_answer_html=highlighted_answer_html,
            final_answer=final_answer,
            applied_rewrites=applied_rewrites,
            rewrite_triggered=rewrite_decision.rewrite_triggered and bool(applied_rewrites),
        )


def build_verification_orchestrator(config: AppConfig | None = None) -> VerificationOrchestrator:
    config = config or get_config()
    client = LMStudioChatClient(config)
    return VerificationOrchestrator(
        config=config,
        claim_extractor=ClaimExtractor(config, client=client),
        retrieval_service=ClaimRetrievalService(config),
        claim_verifier=ClaimVerifier(config, client=client),
        claim_rewriter=ClaimRewriter(config, client=client),
    )
