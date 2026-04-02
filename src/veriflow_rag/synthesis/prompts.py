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


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "prompts" / "answer_synthesis"


def load_prompt_artifacts() -> PromptArtifacts:
    prompt_dir = _prompt_dir()
    system_prompt = (prompt_dir / "system.md").read_text(encoding="utf-8").strip()
    user_template = (prompt_dir / "user_template.md").read_text(encoding="utf-8").strip()
    output_schema = json.loads((prompt_dir / "answer_schema.json").read_text(encoding="utf-8"))
    version = hashlib.sha256(
        (
            system_prompt
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


def render_user_prompt(template: str, *, query: str, evidence_xml: str, schema_json: str) -> str:
    return template.format(
        query=query.strip(),
        evidence=evidence_xml.strip(),
        output_schema=schema_json.strip(),
    )
