from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from veriflow_rag.web.schemas import CorpusRunResponse, DocumentListResponse, DocumentRecord


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    registry = request.app.state.document_registry
    return DocumentListResponse(items=registry.list_documents())


@router.post("/upload", response_model=DocumentListResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
) -> DocumentListResponse:
    registry = request.app.state.document_registry
    uploaded: list[DocumentRecord] = []
    for file in files:
        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            uploaded.append(registry.upload(tmp_path, original_name=file.filename or tmp_path.name))
        finally:
            tmp_path.unlink(missing_ok=True)
    return DocumentListResponse(items=registry.list_documents())


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request) -> None:
    registry = request.app.state.document_registry
    try:
        registry.delete(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
