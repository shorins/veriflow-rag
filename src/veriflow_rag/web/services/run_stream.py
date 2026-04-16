from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import AsyncIterator
from uuid import uuid4

from veriflow_rag.core.config import AppConfig
from veriflow_rag.demo.fault_injection import apply_demo_verification_override, merge_injected_claims
from veriflow_rag.synthesis.models import Citation, SynthesizedAnswer, SynthesisResultBundle
from veriflow_rag.synthesis.service import build_synthesis_service
from veriflow_rag.verification.models import ClaimVerificationResult
from veriflow_rag.verification.orchestrator import build_verification_orchestrator
from veriflow_rag.verification.rewrite import apply_rewrites, select_rewrite_span
from veriflow_rag.web.schemas import RunEvent
from veriflow_rag.web.services.document_registry import DocumentRegistry


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class DraftMessageRecord:
    message_id: str
    query: str
    bundle: SynthesisResultBundle
    draft_model: str
    verification_model: str
    draft_strategy: str
    verification_sensitivity: str
    demo_fault_mode: str
    demo_fault_count: int


class DraftMessageStore:
    def __init__(self) -> None:
        self._items: dict[str, DraftMessageRecord] = {}

    def put(self, record: DraftMessageRecord) -> None:
        self._items[record.message_id] = record

    def get(self, message_id: str) -> DraftMessageRecord:
        item = self._items.get(message_id)
        if item is None:
            raise KeyError(message_id)
        return item


class RunStreamManager:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[RunEvent | None]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def _queue(self, run_id: str) -> asyncio.Queue[RunEvent | None]:
        return self._queues.setdefault(run_id, asyncio.Queue())

    async def emit(self, event: RunEvent) -> None:
        await self._queue(event.run_id).put(event)

    async def close(self, run_id: str) -> None:
        await self._queue(run_id).put(None)

    async def stream(self, run_id: str) -> AsyncIterator[str]:
        queue = self._queue(run_id)
        while True:
            item = await queue.get()
            if item is None:
                break
            payload = item.model_dump(mode="json")
            yield f"event: {item.event_type}\n"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self._queues.pop(run_id, None)
        self._tasks.pop(run_id, None)

    def register(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[run_id] = task

    def has_run(self, run_id: str) -> bool:
        return run_id in self._queues or run_id in self._tasks


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


def _normalize_span(text: str | None) -> str:
    return " ".join((text or "").split()).strip().lower()


async def create_draft_message(
    config: AppConfig,
    message_store: DraftMessageStore,
    *,
    query: str,
) -> DraftMessageRecord:
    service = build_synthesis_service(config)
    bundle = await asyncio.to_thread(service.run_query, query)
    message_id = uuid4().hex
    record = DraftMessageRecord(
        message_id=message_id,
        query=query,
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


async def start_verification_run(
    config: AppConfig,
    message_record: DraftMessageRecord,
    run_manager: RunStreamManager,
) -> str:
    run_id = uuid4().hex

    async def emit(event_type: str, payload: dict, *, message_id: str | None = None) -> None:
        await run_manager.emit(
            RunEvent(
                run_id=run_id,
                event_type=event_type,
                message_id=message_id or message_record.message_id,
                timestamp=utc_now(),
                payload=payload,
            )
        )

    async def worker() -> None:
        orchestrator = build_verification_orchestrator(config)
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
            claims = await asyncio.to_thread(
                orchestrator.claim_extractor.extract_claims,
                message_record.bundle.synthesized_answer.answer,
            )
            claims = merge_injected_claims(
                draft_answer=message_record.bundle.synthesized_answer.answer,
                claims=claims,
                injected_spans=message_record.bundle.synthesized_answer.fault_injection_spans,
            )
            await emit(
                "claims_extracted",
                {
                    "claims": [claim.model_dump(mode="json") for claim in claims],
                },
            )

            claim_results: list[ClaimVerificationResult] = []
            claim_evidence_map: dict[str, list] = {}
            rewritten_spans: dict[str, str] = {}
            for claim in claims:
                await emit(
                    "claim_started",
                    {
                        "claim_id": claim.claim_id,
                        "claim_text": claim.claim_text,
                        "source_span": claim.source_span,
                    },
                )
                evidence_blocks = await asyncio.to_thread(
                    orchestrator.retrieval_service.retrieve_for_claim,
                    claim.claim_text,
                )
                prepared = await asyncio.to_thread(
                    orchestrator.retrieval_service.prepare_claim_evidence,
                    evidence_blocks,
                )
                claim_evidence_map[claim.claim_id] = prepared
                result = await asyncio.to_thread(
                    orchestrator.claim_verifier.verify_claim,
                    claim,
                    prepared,
                )
                result = apply_demo_verification_override(result, message_record.bundle.synthesized_answer.fault_injection_spans)
                result.rewrite_source_span = select_rewrite_span(
                    message_record.bundle.synthesized_answer.answer,
                    result,
                )
                injected_spans = message_record.bundle.synthesized_answer.fault_injection_spans
                normalized_source_span = _normalize_span(result.source_span)
                normalized_claim_text = _normalize_span(result.claim_text)
                if (
                    injected_spans
                    and result.status != "supported"
                    and any(
                        _normalize_span(injected.injected_span) == normalized_source_span
                        or _normalize_span(injected.injected_span) == normalized_claim_text
                        or _normalize_span(injected.injected_span) in normalized_source_span
                        or _normalize_span(injected.injected_span) in normalized_claim_text
                        for injected in injected_spans
                    )
                ):
                    result.rewrite_needed = True
                claim_results.append(result)
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

            forced_rewrite = bool(
                injected_spans and any(result.status != "supported" for result in claim_results)
            )

            for result in claim_results:
                if result.status == "supported":
                    continue
                if forced_rewrite and injected_spans:
                    result.rewrite_needed = True
                if not result.rewrite_needed:
                    continue
                await emit(
                    "rewrite_started",
                    {
                        "claim_id": result.claim_id,
                        "old_span": result.rewrite_source_span or result.source_span,
                    },
                )
                prepared = claim_evidence_map.get(result.claim_id, [])
                rewritten = await asyncio.to_thread(
                    orchestrator.claim_rewriter.rewrite_claim,
                    draft_answer=message_record.bundle.synthesized_answer.answer,
                    claim_result=result,
                    evidence_blocks=prepared,
                )
                if not rewritten:
                    continue
                rewritten_spans[result.claim_id] = rewritten
                await emit(
                    "rewrite_span_typing",
                    {
                        "claim_id": result.claim_id,
                        "old_span": result.rewrite_source_span or result.source_span,
                        "new_span": rewritten,
                        "diff_segments": build_diff_segments(
                            result.rewrite_source_span or result.source_span,
                            rewritten,
                        ),
                    },
                )

            final_answer, applied_rewrites = await asyncio.to_thread(
                apply_rewrites,
                message_record.bundle.synthesized_answer.answer,
                claim_results,
                rewritten_spans,
            )

            await emit(
                "verification_completed",
                {
                    "final_answer": final_answer,
                    "rewrite_triggered": bool(applied_rewrites) or forced_rewrite,
                    "claim_results": [result.model_dump(mode="json") for result in claim_results],
                    "applied_rewrites": [rewrite.model_dump(mode="json") for rewrite in applied_rewrites],
                },
            )
        except Exception as exc:
            await emit(
                "run_error",
                {"message": str(exc)},
            )
        finally:
            await run_manager.close(run_id)

    task = asyncio.create_task(worker())
    run_manager.register(run_id, task)
    return run_id


async def start_corpus_reindex_run(
    config: AppConfig,
    registry: DocumentRegistry,
    run_manager: RunStreamManager,
) -> str:
    run_id = uuid4().hex
    items = registry.list_documents()
    document_ids = [item.document_id for item in items]
    file_names = [item.file_name for item in items]

    async def emit(event_type: str, payload: dict) -> None:
        await run_manager.emit(
            RunEvent(
                run_id=run_id,
                event_type=event_type,
                timestamp=utc_now(),
                payload=payload,
            )
        )

    async def worker() -> None:
        try:
            await emit(
                "corpus_reindex_started",
                {
                    "mode": "reindex",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
            await emit(
                "corpus_reindex_progress",
                {
                    "phase": "sync_registry",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
            await emit(
                "corpus_reindex_progress",
                {
                    "phase": "rebuild_manifest",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
            await emit(
                "corpus_reindex_progress",
                {
                    "phase": "reindex_weaviate",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
            records = await asyncio.to_thread(registry.reindex_corpus)
            await emit(
                "corpus_reindex_progress",
                {
                    "phase": "finalize_registry",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
            await emit(
                "corpus_reindex_completed",
                {
                    "mode": "reindex",
                    "total_documents": len(records),
                    "document_ids": [record.document_id for record in records],
                    "file_names": [record.file_name for record in records],
                    "documents": [record.model_dump(mode="json") for record in records],
                },
            )
        except Exception as exc:
            await emit(
                "corpus_error",
                {
                    "message": str(exc),
                    "mode": "reindex",
                    "total_documents": len(document_ids),
                    "document_ids": document_ids,
                    "file_names": file_names,
                },
            )
        finally:
            await run_manager.close(run_id)

    task = asyncio.create_task(worker())
    run_manager.register(run_id, task)
    return run_id
