from __future__ import annotations

import argparse
import json

from veriflow_rag.synthesis.client import LMStudioClientError
from veriflow_rag.synthesis.service import build_synthesis_service


def run(query: str) -> None:
    service = build_synthesis_service()
    result = service.run_query(query)

    print(f"🔎 query: {query}\n")
    print("=== Selected evidence ===")
    for item in result.evidence_blocks:
        print(f"[{item.evidence_id}] {item.block.file_name} :: {item.block.section_title}")
        print(f"confidence: {item.block.confidence_label}")
        print(item.block.expanded_text[:700].strip())
        print()

    print("=== Structured result ===")
    print(
        json.dumps(
            result.synthesized_answer.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Human-readable summary ===")
    print(result.synthesized_answer.answer)
    for citation in result.synthesized_answer.citations:
        print(
            f"- [{citation.evidence_id}] {citation.file_name} / {citation.section_title}: "
            f"{citation.support}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Что такое информационная система?")
    args = parser.parse_args()
    try:
        run(args.query)
    except LMStudioClientError as exc:
        raise SystemExit(f"❌ {exc}") from exc
