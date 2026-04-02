import unittest

from veriflow_rag.retrieval.pipeline import EvidenceBlock
from veriflow_rag.synthesis.client import LMStudioClientError
from veriflow_rag.synthesis.service import AnswerSynthesisService


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("No fake responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SynthesisServiceTests(unittest.TestCase):
    def _make_block(self, rank: int, confidence: str, text: str) -> EvidenceBlock:
        return EvidenceBlock(
            query="Что такое информационная система?",
            rank=rank,
            retrieval_score=0.8,
            rerank_score=0.7,
            file_name="doc.pdf",
            page_span="1-2",
            section_title="Определение",
            text=text,
            parent_text=text,
            expanded_text=text,
            confidence_label=confidence,
        )

    def _make_config(self):
        return type(
            "Config",
            (),
            {
                "synthesis_top_evidence_k": 4,
                "synthesis_min_confident_evidence": 2,
                "synthesis_model_name": "qwen3.5-9b-instruct",
                "synthesis_temperature": 0.0,
                "synthesis_max_tokens": 700,
                "synthesis_timeout_seconds": 60,
                "synthesis_max_evidence_chars": 1200,
                "lmstudio_base_url": "http://localhost:1234/v1",
                "lmstudio_api_key": "lm-studio",
                "lmstudio_api_mode": "auto",
            },
        )()

    def test_gate_returns_insufficient_context_when_evidence_is_weak(self) -> None:
        config = self._make_config()
        service = AnswerSynthesisService(config=config, client=FakeClient([]))

        result = service.synthesize_answer(
            "Что такое информационная система?",
            [
                self._make_block(1, "medium", "alpha"),
                self._make_block(2, "low", "beta"),
            ],
        )

        self.assertTrue(result.synthesized_answer.insufficient_context)
        self.assertIn("retrieval", result.synthesized_answer.omitted_points[0])

    def test_definition_query_allows_single_high_confidence_evidence(self) -> None:
        config = self._make_config()
        client = FakeClient(
            [
                """{
                  "answer": "Информационная система является интегрированным комплексом ресурсов и средств обработки данных.",
                  "citations": [
                    {
                      "evidence_id": "ev_1",
                      "file_name": "",
                      "section_title": "",
                      "support": "интегрированный комплекс информационных ресурсов"
                    }
                  ],
                  "used_evidence_ids": ["ev_1"],
                  "insufficient_context": false,
                  "omitted_points": []
                }"""
            ]
        )
        service = AnswerSynthesisService(config=config, client=client)

        result = service.synthesize_answer(
            "Что такое информационная система?",
            [
                self._make_block(1, "high", "alpha"),
                self._make_block(2, "low", "beta"),
            ],
        )

        self.assertFalse(result.synthesized_answer.insufficient_context)
        self.assertEqual(client.calls, 1)

    def test_enumeration_query_still_requires_multiple_confident_blocks(self) -> None:
        config = self._make_config()
        service = AnswerSynthesisService(config=config, client=FakeClient([]))

        result = service.synthesize_answer(
            "Какие основные процессы перечислены?",
            [
                self._make_block(1, "high", "alpha"),
                self._make_block(2, "low", "beta"),
            ],
        )

        self.assertTrue(result.synthesized_answer.insufficient_context)

    def test_successful_result_is_normalized_with_local_metadata(self) -> None:
        config = self._make_config()
        client = FakeClient(
            [
                """{
                  "answer": "Информационная система предназначена для сбора, хранения и обработки данных.",
                  "citations": [
                    {
                      "evidence_id": "ev_1",
                      "file_name": "ignored.pdf",
                      "section_title": "ignored",
                      "support": "сбор, хранение и обработка данных"
                    }
                  ],
                  "used_evidence_ids": ["ev_1"],
                  "insufficient_context": false,
                  "omitted_points": []
                }"""
            ]
        )
        service = AnswerSynthesisService(config=config, client=client)

        result = service.synthesize_answer(
            "Что такое информационная система?",
            [
                self._make_block(1, "high", "alpha"),
                self._make_block(2, "medium", "beta"),
            ],
        )

        self.assertFalse(result.synthesized_answer.insufficient_context)
        self.assertEqual(result.synthesized_answer.citations[0].file_name, "doc.pdf")
        self.assertEqual(result.synthesized_answer.used_evidence_ids, ["ev_1"])

    def test_invalid_ids_trigger_retry_and_then_fail_closed(self) -> None:
        config = self._make_config()
        client = FakeClient(
            [
                """{
                  "answer": "Ответ.",
                  "citations": [{"evidence_id": "ev_99", "file_name": "", "section_title": "", "support": "x"}],
                  "used_evidence_ids": ["ev_99"],
                  "insufficient_context": false,
                  "omitted_points": []
                }""",
                """{
                  "answer": "Ответ.",
                  "citations": [{"evidence_id": "ev_99", "file_name": "", "section_title": "", "support": "x"}],
                  "used_evidence_ids": ["ev_99"],
                  "insufficient_context": false,
                  "omitted_points": []
                }""",
            ]
        )
        service = AnswerSynthesisService(config=config, client=client)

        result = service.synthesize_answer(
            "Что такое информационная система?",
            [
                self._make_block(1, "high", "alpha"),
                self._make_block(2, "medium", "beta"),
            ],
        )

        self.assertTrue(result.synthesized_answer.insufficient_context)
        self.assertEqual(client.calls, 2)

    def test_client_error_is_not_silently_swallowed(self) -> None:
        config = self._make_config()
        client = FakeClient([LMStudioClientError("LM Studio is unavailable.")])
        service = AnswerSynthesisService(config=config, client=client)

        with self.assertRaises(LMStudioClientError):
            service.synthesize_answer(
                "Что такое информационная система?",
                [
                    self._make_block(1, "high", "alpha"),
                    self._make_block(2, "medium", "beta"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
