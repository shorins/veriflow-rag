from __future__ import annotations

import argparse
import json

from veriflow_rag.synthesis.service import build_synthesis_service
from veriflow_rag.verification.orchestrator import build_verification_orchestrator


def run(query: str) -> None:
    synthesis_service = build_synthesis_service()
    verification = build_verification_orchestrator()

    draft_bundle = synthesis_service.run_query(query)
    print("=== Draft Answer ===")
    print(draft_bundle.synthesized_answer.answer)
    print()

    result = verification.run(query, draft_bundle)
    print("=== Verification Result ===")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Что такое информационная система?")
    args = parser.parse_args()
    run(args.query)
