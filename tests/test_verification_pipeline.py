import unittest

from veriflow_rag.synthesis.models import PreparedEvidence, SynthesisResultBundle, SynthesizedAnswer
from veriflow_rag.retrieval.pipeline import EvidenceBlock
from veriflow_rag.verification.models import (
    Claim,
    ClaimVerificationResult,
    VerificationRunResult,
)
from veriflow_rag.verification.orchestrator import VerificationOrchestrator
from veriflow_rag.verification.rewrite import apply_rewrites, select_verification_profile, should_trigger_rewrite


class FakeClaimExtractor:
    def __init__(self, claims):
        self.claims = claims

    def extract_claims(self, draft_answer):
        return self.claims


class FakeRetrievalService:
    def __init__(self, prepared):
        self.prepared = prepared

    def retrieve_for_claim(self, claim_text):
        return []

    def prepare_claim_evidence(self, evidence_blocks):
        return self.prepared


class FakeClaimVerifier:
    def __init__(self, results):
        self.results = list(results)

    def verify_claim(self, claim, prepared):
        return self.results.pop(0)


class FakeClaimRewriter:
    def __init__(self, replacements):
        self.replacements = replacements

    def rewrite_claim(self, *, draft_answer, claim_result, evidence_blocks):
        return self.replacements[claim_result.claim_id]


class VerificationPipelineTests(unittest.TestCase):
    def test_should_trigger_rewrite_for_supported_only_returns_false(self) -> None:
        decision = should_trigger_rewrite(
            draft_answer="Alpha. Beta.",
            claim_results=[
                ClaimVerificationResult(
                    claim_id="c1",
                    claim_text="Alpha.",
                    source_span="Alpha.",
                    source_sentence_index=1,
                    status="supported",
                    reason="ok",
                    used_evidence_ids=[],
                    rewrite_needed=False,
                )
            ],
            partial_ratio_threshold=0.2,
            problem_span_ratio_threshold=0.15,
        )
        self.assertFalse(decision.rewrite_triggered)

    def test_should_trigger_rewrite_for_contradicted_returns_true(self) -> None:
        decision = should_trigger_rewrite(
            draft_answer="Alpha. Beta.",
            claim_results=[
                ClaimVerificationResult(
                    claim_id="c1",
                    claim_text="Alpha.",
                    source_span="Alpha.",
                    source_sentence_index=1,
                    status="contradicted",
                    reason="wrong",
                    used_evidence_ids=[],
                    rewrite_needed=True,
                )
            ],
            partial_ratio_threshold=0.2,
            problem_span_ratio_threshold=0.15,
        )
        self.assertTrue(decision.rewrite_triggered)

    def test_demo_verification_profile_is_more_aggressive(self) -> None:
        config = type(
            "Config",
            (),
            {
                "verification_sensitivity": "demo",
                "verification_partial_ratio_threshold": 0.2,
                "verification_problem_span_ratio_threshold": 0.15,
            },
        )()

        profile = select_verification_profile(config)
        self.assertEqual(profile.sensitivity, "demo")
        self.assertTrue(profile.treat_partial_as_rewrite_candidate)
        self.assertLess(profile.partial_ratio_threshold, 0.2)
        self.assertLess(profile.problem_span_ratio_threshold, 0.15)

    def test_apply_rewrites_replaces_non_overlapping_spans(self) -> None:
        draft = "Alpha. Beta."
        claim_results = [
            ClaimVerificationResult(
                claim_id="c1",
                claim_text="Alpha.",
                source_span="Alpha.",
                source_sentence_index=1,
                status="unsupported",
                reason="wrong",
                used_evidence_ids=[],
                rewrite_needed=True,
            ),
            ClaimVerificationResult(
                claim_id="c2",
                claim_text="Beta.",
                source_span="Beta.",
                source_sentence_index=2,
                status="partial",
                reason="partial",
                used_evidence_ids=[],
                rewrite_needed=True,
            ),
        ]
        final_answer, rewrites = apply_rewrites(
            draft_answer=draft,
            claim_results=claim_results,
            rewritten_spans={"c1": "Gamma.", "c2": "Delta."},
        )
        self.assertEqual(final_answer, "Gamma. Delta.")
        self.assertEqual(len(rewrites), 2)

    def test_apply_rewrites_skips_overlapping_spans(self) -> None:
        draft = "Alpha Beta Gamma"
        claim_results = [
            ClaimVerificationResult(
                claim_id="c1",
                claim_text="Alpha Beta",
                source_span="Alpha Beta",
                source_sentence_index=1,
                status="unsupported",
                reason="wrong",
                used_evidence_ids=[],
                rewrite_needed=True,
            ),
            ClaimVerificationResult(
                claim_id="c2",
                claim_text="Beta Gamma",
                source_span="Beta Gamma",
                source_sentence_index=1,
                status="partial",
                reason="partial",
                used_evidence_ids=[],
                rewrite_needed=True,
            ),
        ]
        final_answer, rewrites = apply_rewrites(
            draft_answer=draft,
            claim_results=claim_results,
            rewritten_spans={"c1": "X", "c2": "Y"},
        )
        self.assertEqual(final_answer, "X Gamma")
        self.assertEqual(len(rewrites), 1)

    def test_orchestrator_returns_complete_result(self) -> None:
        claim = Claim(
            claim_id="c1",
            claim_text="Alpha.",
            source_span="Alpha.",
            source_sentence_index=1,
        )
        verification_result = ClaimVerificationResult(
            claim_id="c1",
            claim_text="Alpha.",
            source_span="Alpha.",
            source_sentence_index=1,
            status="unsupported",
            reason="Not supported",
            used_evidence_ids=["ev_1"],
            rewrite_needed=True,
            revised_claim="Gamma.",
        )
        bundle = SynthesisResultBundle(
            query="Q?",
            evidence_blocks=[],
            synthesized_answer=SynthesizedAnswer(
                answer="Alpha. Beta.",
                model_name="qwen",
                prompt_version="v1",
            ),
        )
        prepared = [
            PreparedEvidence(
                evidence_id="ev_1",
                block=EvidenceBlock(
                    query="Q?",
                    rank=1,
                    retrieval_score=1.0,
                    rerank_score=1.0,
                    file_name="doc.pdf",
                    page_span="1",
                    section_title="Section",
                    text="support",
                    parent_text="support",
                    expanded_text="support",
                    confidence_label="high",
                ),
                prompt_text="support",
            )
        ]
        orchestrator = VerificationOrchestrator(
            config=type(
                "Config",
                (),
                {
                    "verification_partial_ratio_threshold": 0.2,
                    "verification_problem_span_ratio_threshold": 0.15,
                },
            )(),
            claim_extractor=FakeClaimExtractor([claim]),
            retrieval_service=FakeRetrievalService(prepared),
            claim_verifier=FakeClaimVerifier([verification_result]),
            claim_rewriter=FakeClaimRewriter({"c1": "Gamma."}),
        )

        result = orchestrator.run("Q?", bundle)
        self.assertIsInstance(result, VerificationRunResult)
        self.assertEqual(result.final_answer, "Gamma. Beta.")
        self.assertTrue(result.rewrite_triggered)
        self.assertEqual(len(result.applied_rewrites), 1)


if __name__ == "__main__":
    unittest.main()
