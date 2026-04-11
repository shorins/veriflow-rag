from __future__ import annotations

from fastapi import APIRouter, Request

from veriflow_rag.web.schemas import CorpusRunResponse
from veriflow_rag.web.services.run_stream import start_corpus_reindex_run


router = APIRouter(prefix="/api/corpus", tags=["corpus"])


@router.post("/reindex/run", response_model=CorpusRunResponse)
async def start_corpus_reindex(request: Request) -> CorpusRunResponse:
    registry = request.app.state.document_registry
    run_manager = request.app.state.run_manager
    run_id = await start_corpus_reindex_run(request.app.state.config, registry, run_manager)
    return CorpusRunResponse(run_id=run_id)
