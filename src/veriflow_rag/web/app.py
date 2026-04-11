from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from veriflow_rag.core.config import get_config
from veriflow_rag.web.routes.chat import router as chat_router
from veriflow_rag.web.routes.corpus import router as corpus_router
from veriflow_rag.web.routes.documents import router as documents_router
from veriflow_rag.web.routes.runs import router as runs_router
from veriflow_rag.web.services.document_registry import DocumentRegistry
from veriflow_rag.web.services.run_stream import DraftMessageStore, RunStreamManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config
    app.state.document_registry = DocumentRegistry(config)
    app.state.message_store = DraftMessageStore()
    app.state.run_manager = RunStreamManager()
    yield


def create_app() -> FastAPI:
    config = get_config()
    app = FastAPI(title="trustRAG Web API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(corpus_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(runs_router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
