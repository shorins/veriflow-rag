import unittest

from veriflow_rag.demo.fault_injection import (
    apply_demo_verification_override,
    inject_demo_faults,
    merge_injected_claims,
)
from veriflow_rag.demo.models import InjectedFaultSpan
from veriflow_rag.verification.models import Claim, ClaimVerificationResult


class DemoFaultInjectionTests(unittest.TestCase):
    def test_off_mode_leaves_answer_untouched(self) -> None:
        result = inject_demo_faults(
            answer="Жизненный цикл включает анализ и проектирование.",
            mode="off",
            count=1,
        )
        self.assertFalse(result.active)
        self.assertIsNone(result.answer)
        self.assertEqual(result.spans, [])

    def test_deterministic_mode_injects_one_supported_looking_fault(self) -> None:
        answer = (
            "Жизненный цикл информационной системы включает анализ и проектирование, "
            "тестирование и сопровождение. "
            "Документирование является важным вспомогательным процессом."
        )
        result = inject_demo_faults(
            answer=answer,
            mode="deterministic",
            count=1,
        )
        self.assertTrue(result.active)
        self.assertEqual(result.count, 1)
        self.assertEqual(len(result.spans), 1)
        self.assertNotEqual(result.answer, answer)
        self.assertIn("visualization", result.summary or "")

    def test_merge_injected_claims_adds_missing_demo_claim(self) -> None:
        claims = [
            Claim(
                claim_id="c1",
                claim_text="Экологический мониторинг помогает принимать решения.",
                source_span="Экологический мониторинг помогает принимать решения.",
                source_sentence_index=1,
            )
        ]
        injected = [
            InjectedFaultSpan(
                claim_id="demo_fault_1",
                fault_type="unsupported_detail",
                original_span="Мониторинг необходим для получения объективных данных.",
                injected_span="Мониторинг необходим для получения объективных данных и полностью заменяет экологическую экспертизу.",
                source_sentence_index=2,
            )
        ]

        merged = merge_injected_claims(
            draft_answer=(
                "Экологический мониторинг помогает принимать решения. "
                "Мониторинг необходим для получения объективных данных и полностью заменяет экологическую экспертизу."
            ),
            claims=claims,
            injected_spans=injected,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[-1].claim_id, "demo_fault_1")
        self.assertIn("полностью заменяет", merged[-1].claim_text)

    def test_demo_override_downgrades_supported_injected_claim(self) -> None:
        result = ClaimVerificationResult(
            claim_id="demo_fault_1",
            claim_text="Мониторинг полностью заменяет экологическую экспертизу.",
            source_span="Мониторинг полностью заменяет экологическую экспертизу.",
            source_sentence_index=2,
            status="supported",
            reason="ok",
            used_evidence_ids=["ev_1"],
            rewrite_needed=False,
        )
        injected = [
            InjectedFaultSpan(
                claim_id="demo_fault_1",
                fault_type="unsupported_detail",
                original_span="Мониторинг помогает принимать решения в области экологии.",
                injected_span="Мониторинг полностью заменяет экологическую экспертизу.",
                source_sentence_index=2,
            )
        ]

        overridden = apply_demo_verification_override(result, injected)

        self.assertEqual(overridden.status, "unsupported")
        self.assertTrue(overridden.rewrite_needed)
        self.assertEqual(
            overridden.revised_claim,
            "Мониторинг помогает принимать решения в области экологии.",
        )


if __name__ == "__main__":
    unittest.main()
