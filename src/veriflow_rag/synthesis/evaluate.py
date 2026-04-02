from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from veriflow_rag.core.config import get_config
from veriflow_rag.synthesis.client import LMStudioClientError
from veriflow_rag.synthesis.models import SynthesizedAnswer
from veriflow_rag.synthesis.service import build_synthesis_service


@dataclass
class SynthesisBenchmarkCase:
    query: str
    expected_file: str | None
    expected_keywords: list[str]
    should_refuse: bool = False


BENCHMARK_CASES = [
    SynthesisBenchmarkCase(
        query="Что такое информационная система?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["информационная система", "комплекс", "процесс"],
    ),
    SynthesisBenchmarkCase(
        query="Какие три фундаментальных процесса работы с данными реализует информационная система?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["сбор", "хран", "обработ"],
    ),
    SynthesisBenchmarkCase(
        query="Что такое декомпозиция при проектировании систем?",
        expected_file="Лекция №1 ИИ конспект.pdf",
        expected_keywords=["декомпози", "разбиение", "част"],
    ),
    SynthesisBenchmarkCase(
        query="Что такое жизненный цикл программного обеспечения?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["жизненный цикл", "последовательность", "эксплуатац"],
    ),
    SynthesisBenchmarkCase(
        query="Что определяет стандарт ISO/IEC 12207?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["структур", "жизненного цикла", "процесс"],
    ),
    SynthesisBenchmarkCase(
        query="Какие основные процессы жизненного цикла перечислены в ISO/IEC 12207?",
        expected_file="Лекция №2 ИИ конспект.pdf",
        expected_keywords=["acquisition", "development", "maintenance"],
    ),
    SynthesisBenchmarkCase(
        query="Какая столица Франции?",
        expected_file=None,
        expected_keywords=[],
        should_refuse=True,
    ),
    SynthesisBenchmarkCase(
        query="Как лечат пневмонию у взрослых?",
        expected_file=None,
        expected_keywords=[],
        should_refuse=True,
    ),
]


def assess_answer(case: SynthesisBenchmarkCase, answer: SynthesizedAnswer) -> str:
    if case.should_refuse:
        return "релевантно" if answer.insufficient_context else "нерелевантно"

    if answer.insufficient_context:
        return "нерелевантно"

    text = answer.answer.lower()
    keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in text)
    file_hits = sum(1 for citation in answer.citations if citation.file_name == case.expected_file)

    if file_hits >= 1 and keyword_hits >= max(2, len(case.expected_keywords) - 1):
        return "релевантно"
    if file_hits >= 1 and keyword_hits >= 1:
        return "частично"
    return "нерелевантно"


def render_answer(answer: SynthesizedAnswer) -> str:
    citations = "\n".join(
        f"- [{citation.evidence_id}] {citation.file_name} / {citation.section_title}: {citation.support}"
        for citation in answer.citations
    ) or "- _нет citations_"
    return (
        f"Ответ: {answer.answer}\n\n"
        f"Недостаточно контекста: {answer.insufficient_context}\n\n"
        f"Citations:\n{citations}"
    )


def _build_markdown(results: list[dict], success_count: int, total_cases: int) -> str:
    markdown_lines = [
        "# Synthesis benchmark",
        "",
        "Сравнение constrained answer synthesis на тестовых вопросах по текущему retrieval baseline.",
        "",
    ]

    for item in results:
        case = item["case"]
        markdown_lines.extend(
            [
                f"## {case['query']}",
                "",
                f"- Expected file: `{case['expected_file']}`",
                f"- Should refuse: `{case['should_refuse']}`",
                f"- Assessment: `{item['assessment']}`",
                "",
                render_answer(SynthesizedAnswer.model_validate(item["answer"])),
                "",
            ]
        )

    markdown_lines.extend(
        [
            "## Итог",
            "",
            f"- Обработано кейсов: `{len(results)}` из `{total_cases}`.",
            f"- Успешных кейсов: `{success_count}` из `{total_cases}`.",
            "- Основные критерии: groundedness, корректные citations и честный отказ при слабом контексте.",
        ]
    )
    return "\n".join(markdown_lines)


def _persist_progress(config, results: list[dict], success_count: int, total_cases: int) -> None:
    config.synthesis_benchmark_json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config.synthesis_benchmark_report_path.write_text(
        _build_markdown(results, success_count, total_cases),
        encoding="utf-8",
    )


def run_benchmark() -> None:
    config = get_config()
    service = build_synthesis_service()

    results = []
    success_count = 0

    for index, case in enumerate(BENCHMARK_CASES, start=1):
        print(f"▶️  [{index}/{len(BENCHMARK_CASES)}] Retrieval + synthesis: {case.query}", flush=True)
        bundle = service.run_query(case.query)
        assessment = assess_answer(case, bundle.synthesized_answer)
        if assessment == "релевантно":
            success_count += 1

        result_item = {
            "case": asdict(case),
            "assessment": assessment,
            "answer": bundle.synthesized_answer.model_dump(),
        }
        results.append(result_item)

        _persist_progress(
            config=config,
            results=results,
            success_count=success_count,
            total_cases=len(BENCHMARK_CASES),
        )
        print(
            f"✅ [{index}/{len(BENCHMARK_CASES)}] Saved result: assessment={assessment}, "
            f"insufficient_context={bundle.synthesized_answer.insufficient_context}",
            flush=True,
        )

    print(f"✅ Synthesis benchmark report saved to {config.synthesis_benchmark_report_path}")
    print(f"✅ Synthesis benchmark JSON saved to {config.synthesis_benchmark_json_path}")


if __name__ == "__main__":
    try:
        run_benchmark()
    except LMStudioClientError as exc:
        raise SystemExit(f"❌ {exc}") from exc
