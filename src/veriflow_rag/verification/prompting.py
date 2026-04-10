from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptArtifacts:
    system_prompt: str
    user_template: str
    output_schema: dict
    version: str


def _prompt_dir(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / name


def load_prompt_artifacts(name: str) -> PromptArtifacts:
    prompt_dir = _prompt_dir(name)
    system_prompt = (prompt_dir / "system.md").read_text(encoding="utf-8").strip()
    user_template = (prompt_dir / "user_template.md").read_text(encoding="utf-8").strip()
    output_schema = json.loads((prompt_dir / "schema.json").read_text(encoding="utf-8"))
    version = hashlib.sha256(
        (
            name
            + "\n---\n"
            + system_prompt
            + "\n---\n"
            + user_template
            + "\n---\n"
            + json.dumps(output_schema, ensure_ascii=False, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()[:12]
    return PromptArtifacts(
        system_prompt=system_prompt,
        user_template=user_template,
        output_schema=output_schema,
        version=version,
    )


def render_user_prompt(template: str, **kwargs: str) -> str:
    normalized = {key: value.strip() if isinstance(value, str) else value for key, value in kwargs.items()}
    return template.format(**normalized)
