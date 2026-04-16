from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
from uuid import uuid4

from veriflow_rag.core.config import AppConfig
from veriflow_rag.synthesis.models import Citation, SynthesisResultBundle, SynthesizedAnswer
from veriflow_rag.verification.models import AppliedRewrite, Claim, ClaimVerificationResult
from veriflow_rag.web.schemas import RunEvent

if TYPE_CHECKING:
    from veriflow_rag.web.services.run_stream import DraftMessageRecord, DraftMessageStore, RunStreamManager


UI_TEST_QUERY = "Зачем нужен экологический мониторинг и какие задачи он решает?"
UI_TEST_DRAFT_ANSWER = (
    "Экологический мониторинг - это система регулярных наблюдений за состоянием окружающей среды.\n\n"
    "Он помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы. "
    "Кроме того, экологический мониторинг полностью заменяет экологическую экспертизу при принятии решений.\n\n"
    "Полученные данные используются для оценки рисков и подготовки природоохранных мер."
)
UI_TEST_GROUNDED_ANSWER = (
    "Экологический мониторинг - это система регулярных наблюдений за состоянием окружающей среды.\n\n"
    "Он помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы. "
    "Кроме того, экологический мониторинг предоставляет данные для экологической экспертизы и принятия решений.\n\n"
    "Полученные данные используются для оценки рисков и подготовки природоохранных мер."
)
UI_TEST_PROBLEM_SPAN = "Кроме того, экологический мониторинг полностью заменяет экологическую экспертизу при принятии решений."
UI_TEST_REWRITTEN_SPAN = "Кроме того, экологический мониторинг предоставляет данные для экологической экспертизы и принятия решений."


def utc_now() -> datetime:
    return datetime.now(UTC)


def build_diff_segments(old_text: str, new_text: str) -> list[dict[str, str]]:
    matcher = SequenceMatcher(a=old_text, b=new_text)
    segments: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append({"kind": "equal", "value": old_text[i1:i2]})
        elif tag == "delete":
            segments.append({"kind": "delete", "value": old_text[i1:i2]})
        elif tag == "insert":
            segments.append({"kind": "insert", "value": new_text[j1:j2]})
        else:
            deleted = old_text[i1:i2]
            inserted = new_text[j1:j2]
            if deleted:
                segments.append({"kind": "delete", "value": deleted})
            if inserted:
                segments.append({"kind": "insert", "value": inserted})
    return segments


def build_ui_test_bundle(config: AppConfig) -> SynthesisResultBundle:
    answer = SynthesizedAnswer(
        answer=UI_TEST_DRAFT_ANSWER,
        grounded_answer=UI_TEST_GROUNDED_ANSWER,
        citations=[
            Citation(
                evidence_id="mock_ev_1",
                file_name="ui-test-fixture.pdf",
                section_title="Экологический мониторинг",
                support="Мониторинг используется для наблюдения за состоянием среды и выявления загрязнений.",
            ),
            Citation(
                evidence_id="mock_ev_2",
                file_name="ui-test-fixture.pdf",
                section_title="Использование данных мониторинга",
                support="Данные мониторинга применяются при экологической оценке и подготовке природоохранных решений.",
            ),
        ],
        used_evidence_ids=["mock_ev_1", "mock_ev_2"],
        insufficient_context=False,
        omitted_points=[],
        answer_depth="detailed",
        model_name=config.draft_model_name,
        prompt_version="ui-test-default",
    )
    return SynthesisResultBundle(
        query=UI_TEST_QUERY,
        evidence_blocks=[],
        synthesized_answer=answer,
    )


def create_ui_test_draft_record(
    config: AppConfig,
    message_store: "DraftMessageStore",
    *,
    query: str,
) -> "DraftMessageRecord":
    from veriflow_rag.web.services.run_stream import DraftMessageRecord

    message_id = uuid4().hex
    bundle = build_ui_test_bundle(config)
    record = DraftMessageRecord(
        message_id=message_id,
        query=UI_TEST_QUERY if query.strip() else bundle.query,
        bundle=bundle,
        draft_model=config.draft_model_name,
        verification_model=config.verification_model_name,
        draft_strategy=config.draft_strategy,
        verification_sensitivity=config.verification_sensitivity,
        demo_fault_mode=getattr(config, "demo_fault_mode", "off"),
        demo_fault_count=getattr(config, "demo_fault_count", 1),
    )
    message_store.put(record)
    return record


def _ui_test_claims() -> list[Claim]:
    return [
        Claim(
            claim_id="c1",
            claim_text="Экологический мониторинг помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы.",
            source_span="Он помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы.",
            source_sentence_index=2,
        ),
        Claim(
            claim_id="c2",
            claim_text="Экологический мониторинг полностью заменяет экологическую экспертизу при принятии решений.",
            source_span=UI_TEST_PROBLEM_SPAN,
            source_sentence_index=3,
        ),
        Claim(
            claim_id="c3",
            claim_text="Полученные данные используются для оценки рисков и подготовки природоохранных мер.",
            source_span="Полученные данные используются для оценки рисков и подготовки природоохранных мер.",
            source_sentence_index=4,
        ),
    ]


def _ui_test_claim_results() -> list[ClaimVerificationResult]:
    return [
        ClaimVerificationResult(
            claim_id="c1",
            claim_text="Экологический мониторинг помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы.",
            source_span="Он помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы.",
            rewrite_source_span="Он помогает выявлять загрязнение и отслеживать изменения воздуха, воды и почвы.",
            source_sentence_index=2,
            status="supported",
            reason="Этот фрагмент соответствует evidence о задачах мониторинга и наблюдении за состоянием среды.",
            used_evidence_ids=["mock_ev_1"],
            rewrite_needed=False,
        ),
        ClaimVerificationResult(
            claim_id="c2",
            claim_text="Экологический мониторинг полностью заменяет экологическую экспертизу при принятии решений.",
            source_span=UI_TEST_PROBLEM_SPAN,
            rewrite_source_span=UI_TEST_PROBLEM_SPAN,
            source_sentence_index=3,
            status="unsupported",
            reason=(
                "В материалах мониторинг описывается как источник данных для экологической оценки "
                "и принятия решений, но не как полная замена экологической экспертизы."
            ),
            used_evidence_ids=["mock_ev_1", "mock_ev_2"],
            rewrite_needed=True,
            revised_claim=UI_TEST_REWRITTEN_SPAN,
        ),
        ClaimVerificationResult(
            claim_id="c3",
            claim_text="Полученные данные используются для оценки рисков и подготовки природоохранных мер.",
            source_span="Полученные данные используются для оценки рисков и подготовки природоохранных мер.",
            rewrite_source_span="Полученные данные используются для оценки рисков и подготовки природоохранных мер.",
            source_sentence_index=4,
            status="supported",
            reason="Этот фрагмент поддержан evidence о применении данных мониторинга в природоохранных решениях.",
            used_evidence_ids=["mock_ev_2"],
            rewrite_needed=False,
        ),
    ]


def _ui_test_applied_rewrites() -> list[AppliedRewrite]:
    return [
        AppliedRewrite(
            claim_id="c2",
            old_span=UI_TEST_PROBLEM_SPAN,
            new_span=UI_TEST_REWRITTEN_SPAN,
            rewrite_source_span=UI_TEST_PROBLEM_SPAN,
            status_before="unsupported",
        )
    ]


def _build_ui_test_final_answer() -> str:
    return UI_TEST_DRAFT_ANSWER.replace(UI_TEST_PROBLEM_SPAN, UI_TEST_REWRITTEN_SPAN, 1)


async def start_ui_test_verification_run(
    message_record: "DraftMessageRecord",
    run_manager: "RunStreamManager",
) -> str:
    run_id = uuid4().hex
    claims = _ui_test_claims()
    claim_results = _ui_test_claim_results()
    final_answer = _build_ui_test_final_answer()
    applied_rewrites = _ui_test_applied_rewrites()

    async def emit(event_type: str, payload: dict) -> None:
        await run_manager.emit(
            RunEvent(
                run_id=run_id,
                event_type=event_type,
                message_id=message_record.message_id,
                timestamp=utc_now(),
                payload=payload,
            )
        )

    async def worker() -> None:
        try:
            await emit(
                "verification_started",
                {
                    "query": message_record.query,
                    "draft_answer": message_record.bundle.synthesized_answer.answer,
                    "grounded_answer": message_record.bundle.synthesized_answer.grounded_answer,
                    "draft_model": message_record.draft_model,
                    "verification_model": message_record.verification_model,
                    "demo_fault_mode": message_record.demo_fault_mode,
                },
            )
            await asyncio.sleep(0.15)
            await emit(
                "claims_extracted",
                {"claims": [claim.model_dump(mode="json") for claim in claims]},
            )

            for claim, result in zip(claims, claim_results, strict=True):
                await asyncio.sleep(0.45)
                await emit(
                    "claim_started",
                    {
                        "claim_id": claim.claim_id,
                        "claim_text": claim.claim_text,
                        "source_span": claim.source_span,
                    },
                )
                await asyncio.sleep(0.85)
                await emit(
                    f"claim_{result.status}",
                    {
                        "claim_id": result.claim_id,
                        "claim_text": result.claim_text,
                        "status": result.status,
                        "reason": result.reason,
                        "used_evidence_ids": result.used_evidence_ids,
                        "rewrite_needed": result.rewrite_needed,
                        "source_span": result.source_span,
                        "rewrite_source_span": result.rewrite_source_span,
                        "revised_claim": result.revised_claim,
                    },
                )

            await asyncio.sleep(0.8)
            await emit(
                "rewrite_started",
                {
                    "claim_id": "c2",
                    "old_span": UI_TEST_PROBLEM_SPAN,
                },
            )
            await asyncio.sleep(0.45)
            await emit(
                "rewrite_span_typing",
                {
                    "claim_id": "c2",
                    "old_span": UI_TEST_PROBLEM_SPAN,
                    "new_span": UI_TEST_REWRITTEN_SPAN,
                    "diff_segments": build_diff_segments(UI_TEST_PROBLEM_SPAN, UI_TEST_REWRITTEN_SPAN),
                },
            )
            await asyncio.sleep(0.2)
            await emit(
                "verification_completed",
                {
                    "final_answer": final_answer,
                    "rewrite_triggered": True,
                    "claim_results": [result.model_dump(mode="json") for result in claim_results],
                    "applied_rewrites": [rewrite.model_dump(mode="json") for rewrite in applied_rewrites],
                },
            )
        except Exception as exc:
            await emit("run_error", {"message": str(exc)})
        finally:
            await run_manager.close(run_id)

    task = asyncio.create_task(worker())
    run_manager.register(run_id, task)
    return run_id
