from __future__ import annotations

import json
import re


def extract_json_payload(text: str) -> dict | list:
    candidate = text.strip()
    if not candidate:
        raise json.JSONDecodeError("Empty response", text, 0)

    parsers = [
        candidate,
        _strip_code_fence(candidate),
    ]
    parsers.extend(_extract_balanced_candidates(candidate))

    seen: set[str] = set()
    last_error: json.JSONDecodeError | None = None
    for parser_input in parsers:
        parser_input = parser_input.strip()
        if not parser_input or parser_input in seen:
            continue
        seen.add(parser_input)
        try:
            return json.loads(parser_input)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Unable to extract JSON payload", text, 0)


def _strip_code_fence(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_balanced_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates
