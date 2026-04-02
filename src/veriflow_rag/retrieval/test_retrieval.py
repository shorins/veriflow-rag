from __future__ import annotations

import argparse

from veriflow_rag.retrieval.pipeline import build_retriever


def test_hybrid_search(query_text: str, use_legacy: bool = False) -> None:
    retriever = build_retriever(use_legacy=use_legacy)
    results = retriever.search(query_text)
    if not results:
        print("❌ No retrieval results.")
        return

    mode = "legacy" if use_legacy else "baseline"
    print(f"🔎 {mode} retrieval for query: {query_text}\n")
    for result in results[:5]:
        print(f"--- Result #{result.rank} ---")
        print(f"file: {result.file_name}")
        print(f"section: {result.section_title}")
        print(f"retrieval_score: {result.retrieval_score}")
        print(f"rerank_score: {result.rerank_score}")
        print(f"confidence: {result.confidence_label}")
        print(result.expanded_text[:1200])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Что такое информационная система?")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    test_hybrid_search(args.query, use_legacy=args.legacy)
