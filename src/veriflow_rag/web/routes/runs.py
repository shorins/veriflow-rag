from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("/{run_id}/events")
async def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    run_manager = request.app.state.run_manager
    return StreamingResponse(
        run_manager.stream(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

