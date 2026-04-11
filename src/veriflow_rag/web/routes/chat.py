from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from veriflow_rag.core.config import get_config
from veriflow_rag.web.schemas import DraftRequest, DraftResponse, StartVerificationRequest, StartVerificationResponse
from veriflow_rag.web.services.run_stream import create_draft_message, start_verification_run


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/draft", response_model=DraftResponse)
async def create_draft(request: Request, payload: DraftRequest) -> DraftResponse:
    config = (
        get_config()
        .with_draft_model(payload.draft_model)
        .with_verification_model(payload.verification_model)
        .with_draft_strategy(payload.draft_strategy)
        .with_verification_sensitivity(payload.verification_sensitivity)
    )
    message_store = request.app.state.message_store
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

    config = (
        get_config()
        .with_draft_model(payload.draft_model)
        .with_verification_model(payload.verification_model)
        .with_draft_strategy(payload.draft_strategy)
        .with_verification_sensitivity(payload.verification_sensitivity)
    )
    run_id = await start_verification_run(config, message_record, run_manager)
    return StartVerificationResponse(run_id=run_id, message_id=message_id)

