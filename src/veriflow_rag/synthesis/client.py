from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from veriflow_rag.core.config import AppConfig


class LMStudioClientError(RuntimeError):
    """Raised when LM Studio cannot be reached or returns an invalid response."""


@dataclass
class LMStudioChatClient:
    config: AppConfig

    def _resolve_mode(self) -> str:
        mode = self.config.lmstudio_api_mode.lower().strip()
        if mode in {"openai", "native"}:
            return mode
        parsed = urlparse(self.config.lmstudio_base_url)
        if parsed.path.rstrip("/").endswith("/v1"):
            return "openai"
        return "native"

    def _openai_url(self) -> str:
        base = self.config.lmstudio_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _native_url(self) -> str:
        base = self.config.lmstudio_base_url.rstrip("/")
        return f"{base}/api/v1/chat"

    def chat_json(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:
        mode = self._resolve_mode()
        if mode == "openai":
            url = self._openai_url()
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_synthesis",
                        "schema": output_schema,
                    },
                },
            }
        else:
            url = self._native_url()
            payload = {
                "model": model_name,
                "system_prompt": system_prompt,
                "input": user_prompt,
            }

        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.lmstudio_api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LMStudioClientError(
                f"LM Studio returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise LMStudioClientError(
                "LM Studio is unavailable. Start the local server in LM Studio and ensure "
                f"the API is reachable at {self.config.lmstudio_base_url}."
            ) from exc

        if mode == "openai":
            try:
                return body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LMStudioClientError(
                    f"LM Studio response does not contain chat content: {body}"
                ) from exc

        for key in ("output", "response", "text", "content"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, str) and content.strip():
                            parts.append(content)
                if parts:
                    return "\n".join(parts)

        if isinstance(body.get("messages"), list):
            for message in reversed(body["messages"]):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LMStudioClientError(
                f"LM Studio response does not contain chat content: {body}"
            ) from exc
