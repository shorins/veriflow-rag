from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from veriflow_rag.core.config import AppConfig
from veriflow_rag.web.schemas import DraftRequest, DraftResponse, StartVerificationRequest, StartVerificationResponse
from veriflow_rag.web.services.run_stream import create_draft_message, start_verification_run
from veriflow_rag.web.services.ui_test_mock import create_ui_test_draft_record, start_ui_test_verification_run


router = APIRouter(prefix="/api/chat", tags=["chat"])


def _config_from_request(request: Request, payload: DraftRequest | StartVerificationRequest) -> AppConfig:
    return (
        request.app.state.config
        .with_draft_model(payload.draft_model)
        .with_verification_model(payload.verification_model)
        .with_draft_strategy(payload.draft_strategy)
        .with_verification_sensitivity(payload.verification_sensitivity)
        .with_demo_fault_mode(payload.demo_fault_mode)
        .with_demo_fault_count(payload.demo_fault_count)
    )


def _is_ui_test_request(request: Request) -> bool:
    config = request.app.state.config
    if not getattr(config, "web_ui_test_mode", False):
        return False
    query_flag = request.query_params.get("ui_test", "").strip().lower()
    header_flag = request.headers.get("X-TrustRAG-UI-Test", "").strip().lower()
    return query_flag in {"1", "true", "yes"} or header_flag in {"1", "true", "yes"}


@router.post("/draft", response_model=DraftResponse)
async def create_draft(request: Request, payload: DraftRequest) -> DraftResponse:
    config = _config_from_request(request, payload)
    message_store = request.app.state.message_store
    if _is_ui_test_request(request):
        record = create_ui_test_draft_record(config, message_store, query=payload.query)
    else:
        record = await create_draft_message(config, message_store, query=payload.query)
    answer = record.bundle.synthesized_answer
    return DraftResponse(
        message_id=record.message_id,
        query=record.query,
        draft_answer=answer.answer,
        answer_depth=answer.answer_depth,
        insufficient_context=answer.insufficient_context,
        citations=answer.citations,
        draft_model=record.draft_model,
        verification_model=record.verification_model,
        draft_strategy=record.draft_strategy,
        verification_sensitivity=record.verification_sensitivity,
        grounded_answer=answer.grounded_answer,
        fault_injection_active=answer.fault_injection_active,
        demo_fault_mode=answer.fault_injection_mode,
        demo_fault_count=answer.fault_injection_count,
        fault_injection_summary=answer.fault_injection_summary,
        fault_injection_spans=answer.fault_injection_spans,
    )


@router.post("/{message_id}/verify", response_model=StartVerificationResponse)
async def start_verification(
    message_id: str,
    request: Request,
    payload: StartVerificationRequest,
) -> StartVerificationResponse:
    message_store = request.app.state.message_store
    run_manager = request.app.state.run_manager
    try:
        message_record = message_store.get(message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Message not found") from exc

    config = _config_from_request(request, payload)
    if _is_ui_test_request(request):
        run_id = await start_ui_test_verification_run(message_record, run_manager)
    else:
        run_id = await start_verification_run(config, message_record, run_manager)
    return StartVerificationResponse(run_id=run_id, message_id=message_id)
