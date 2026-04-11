import unittest

from veriflow_rag.demo.fault_injection import inject_demo_faults


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


if __name__ == "__main__":
    unittest.main()
