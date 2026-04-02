from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from veriflow_rag.core.config import get_config
from veriflow_rag.retrieval.pipeline import EvidenceBlock, build_retriever


@dataclass
class RetrievalBenchmarkCase:
    query: str
    expected_file: str
    expected_keywords: list[str]


BENCHMARK_CASES = [
    RetrievalBenchmarkCase(
        query="Что такое информационная система?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["информационная система", "интегрированный комплекс", "процессов организации"],
    ),
    RetrievalBenchmarkCase(
        query="Какие три фундаментальных процесса работы с данными реализует информационная система?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["сбор", "хранение", "обработку"],
    ),
    RetrievalBenchmarkCase(
        query="Что такое декомпозиция при проектировании систем?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["декомпозиции", "разбиение сложной системы", "независимые части"],
    ),
    RetrievalBenchmarkCase(
        query="Что такое жизненный цикл программного обеспечения?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["жизненный цикл", "формализованную последовательность процессов", "вывода системы из эксплуатации"],
    ),
    RetrievalBenchmarkCase(
        query="Что определяет стандарт ISO/IEC 12207?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["структуру жизненного цикла", "совокупность процессов разработки", "взаимодействие между участниками"],
    ),
    RetrievalBenchmarkCase(
        query="Какие основные процессы жизненного цикла перечислены в ISO/IEC 12207?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["acquisition", "development", "maintenance"],
    ),
]


def assess_block(case: RetrievalBenchmarkCase, block: EvidenceBlock | None) -> str:
    if block is None:
        return "нерелевантно"

    keyword_hits = sum(
        1 for keyword in case.expected_keywords if keyword.lower() in block.expanded_text.lower()
    )
    file_ok = block.file_name == case.expected_file
    if file_ok and keyword_hits >= max(2, len(case.expected_keywords) - 1):
        return "релевантно"
    if file_ok and keyword_hits >= 1:
        return "частично"
    return "нерелевантно"


def render_block(block: EvidenceBlock | None) -> str:
    if block is None:
        return "_нет результата_"
    return (
        f"Файл: {block.file_name}\n\n"
        f"Раздел: {block.section_title}\n\n"
        f"Confidence: {block.confidence_label}\n\n"
        f"Фрагмент:\n{block.expanded_text[:1500].strip()}"
    )


def run_benchmark() -> None:
    config = get_config()
    legacy = build_retriever(use_legacy=True)
    baseline = build_retriever(use_legacy=False)

    results = []
    markdown_lines = [
        "# Retrieval benchmark",
        "",
        "Сравнение legacy baseline и нового retrieval-first baseline на тестовых PDF из `data/`.",
        "",
    ]

    baseline_better = 0
    for case in BENCHMARK_CASES:
        legacy_results = legacy.search(case.query)
        baseline_results = baseline.search(case.query)
        legacy_top = legacy_results[0] if legacy_results else None
        baseline_top = baseline_results[0] if baseline_results else None

        legacy_assessment = assess_block(case, legacy_top)
        baseline_assessment = assess_block(case, baseline_top)

        if baseline_assessment == "релевантно" and legacy_assessment != "релевантно":
            baseline_better += 1
        elif baseline_assessment == "частично" and legacy_assessment == "нерелевантно":
            baseline_better += 1

        results.append(
            {
                "case": asdict(case),
                "legacy_assessment": legacy_assessment,
                "baseline_assessment": baseline_assessment,
                "legacy_top": asdict(legacy_top) if legacy_top else None,
                "baseline_top": asdict(baseline_top) if baseline_top else None,
            }
        )

        markdown_lines.extend(
            [
                f"## {case.query}",
                "",
                f"- Ожидаемый файл: `{case.expected_file}`",
                f"- Legacy: `{legacy_assessment}`",
                f"- Baseline: `{baseline_assessment}`",
                "",
                "### Legacy top-1",
                "",
                render_block(legacy_top),
                "",
                "### New baseline top-1",
                "",
                render_block(baseline_top),
                "",
            ]
        )

    markdown_lines.extend(
        [
            "## Итог",
            "",
            f"- Новый baseline оказался лучше legacy на `{baseline_better}` из `{len(BENCHMARK_CASES)}` запросов.",
            "- Если baseline всё ещё промахивается, это нужно интерпретировать как проблему конкретного PDF parsing или границ секций, а не как повод возвращаться к плоскому chunking.",
        ]
    )

    config.benchmark_json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config.benchmark_report_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print(f"✅ Benchmark report saved to {config.benchmark_report_path}")
    print(f"✅ Benchmark JSON saved to {config.benchmark_json_path}")


if __name__ == "__main__":
    run_benchmark()
