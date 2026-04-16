import asyncio
import unittest

from veriflow_rag.core.config import get_config
from veriflow_rag.web.services.run_stream import DraftMessageStore, RunStreamManager
from veriflow_rag.web.services.ui_test_mock import (
    UI_TEST_DRAFT_ANSWER,
    UI_TEST_QUERY,
    UI_TEST_REWRITTEN_SPAN,
    build_ui_test_bundle,
    create_ui_test_draft_record,
    start_ui_test_verification_run,
)


class UITestMockTests(unittest.TestCase):
    def test_build_ui_test_bundle_returns_canned_answer(self) -> None:
        bundle = build_ui_test_bundle(get_config())
        self.assertEqual(bundle.query, UI_TEST_QUERY)
        self.assertEqual(bundle.synthesized_answer.answer, UI_TEST_DRAFT_ANSWER)
        self.assertIn("mock_ev_1", bundle.synthesized_answer.used_evidence_ids)

    def test_create_ui_test_draft_record_uses_canned_query(self) -> None:
        store = DraftMessageStore()
        record = create_ui_test_draft_record(get_config(), store, query="Любой вопрос")
        self.assertEqual(record.query, UI_TEST_QUERY)
        self.assertEqual(store.get(record.message_id).message_id, record.message_id)

    def test_start_ui_test_verification_run_emits_expected_event_order(self) -> None:
        async def collect() -> list[str]:
            store = DraftMessageStore()
            manager = RunStreamManager()
            record = create_ui_test_draft_record(get_config(), store, query=UI_TEST_QUERY)
            run_id = await start_ui_test_verification_run(record, manager)
            events: list[str] = []
            async for chunk in manager.stream(run_id):
                if chunk.startswith("event: "):
                    events.append(chunk.removeprefix("event: ").strip())
            return events

        events = asyncio.run(collect())
        self.assertEqual(
            events,
            [
                "verification_started",
                "claims_extracted",
                "claim_started",
                "claim_supported",
                "claim_started",
                "claim_unsupported",
                "claim_started",
                "claim_supported",
                "rewrite_started",
                "rewrite_span_typing",
                "verification_completed",
            ],
        )
        self.assertTrue(any(event == "rewrite_span_typing" for event in events))
        self.assertIn(UI_TEST_REWRITTEN_SPAN, build_ui_test_bundle(get_config()).synthesized_answer.grounded_answer)


if __name__ == "__main__":
    unittest.main()
