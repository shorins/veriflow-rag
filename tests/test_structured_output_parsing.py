import unittest

from pydantic import BaseModel

from veriflow_rag.verification.llm import StructuredLLMRunner


class DummyResponse(BaseModel):
    claims: list[dict]


class FakeClient:
    def __init__(self, response: str):
        self.response = response

    def chat_json(self, **kwargs):
        return self.response


class StructuredOutputParsingTests(unittest.TestCase):
    def test_runner_accepts_markdown_fenced_json(self) -> None:
        runner = StructuredLLMRunner(
            config=type("Config", (), {})(),
            client=FakeClient(
                """```json
{
  "claims": [
    {
      "claim_id": "1",
      "claim_text": "Alpha",
      "source_span": "Alpha",
      "source_sentence_index": 1
    }
  ]
}
```"""
            ),
        )

        result = runner.run(
            model_name="dummy",
            system_prompt="system",
            user_prompt="user",
            output_schema={"type": "object"},
            temperature=0.0,
            max_tokens=128,
            timeout_seconds=30,
            response_model=DummyResponse,
        )

        self.assertEqual(len(result.claims), 1)


if __name__ == "__main__":
    unittest.main()
