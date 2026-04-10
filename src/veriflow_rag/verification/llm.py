from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from veriflow_rag.core.config import AppConfig
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.tools.json_extract import extract_json_payload


T = TypeVar("T", bound=BaseModel)


class StructuredLLMRunner:
    def __init__(self, config: AppConfig, client: LMStudioChatClient | None = None) -> None:
        self.config = config
        self.client = client or LMStudioChatClient(config)

    def run(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        response_model: type[T],
    ) -> T:
        last_error: Exception | None = None
        reminder = "\n\n<format_reminder>\nReturn only valid JSON that strictly follows the schema.\n</format_reminder>"
        for retry in (False, True):
            prompt = user_prompt + (reminder if retry else "")
            try:
                raw_text = self.client.chat_json(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    output_schema=output_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                payload = extract_json_payload(raw_text)
                return response_model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"Structured output validation failed: {last_error}")
