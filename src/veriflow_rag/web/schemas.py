from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from veriflow_rag.core.config import DraftStrategy, VerificationSensitivity
from veriflow_rag.synthesis.models import AnswerDepth, Citation
from veriflow_rag.verification.models import ClaimStatus


DocumentStatus = Literal["uploaded", "indexed", "stale", "error"]


class DocumentRecord(BaseModel):
    document_id: str
    file_name: str
    stored_path: str
    size_bytes: int
    uploaded_at: datetime
    status: DocumentStatus
    last_parsed_at: datetime | None = None
    last_indexed_at: datetime | None = None
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    items: list[DocumentRecord] = Field(default_factory=list)


class CorpusRunResponse(BaseModel):
    run_id: str


class DraftRequest(BaseModel):
    query: str
    draft_model: str
    verification_model: str
    draft_strategy: DraftStrategy
    verification_sensitivity: VerificationSensitivity


class DraftResponse(BaseModel):
    message_id: str
    query: str
    draft_answer: str
    answer_depth: AnswerDepth
    insufficient_context: bool
    citations: list[Citation] = Field(default_factory=list)
    draft_model: str
    verification_model: str
    draft_strategy: DraftStrategy
    verification_sensitivity: VerificationSensitivity


class StartVerificationRequest(BaseModel):
    query: str
    draft_answer: str
    answer_depth: AnswerDepth
    draft_model: str
    verification_model: str
    draft_strategy: DraftStrategy
    verification_sensitivity: VerificationSensitivity


class StartVerificationResponse(BaseModel):
    run_id: str
    message_id: str


class RunEvent(BaseModel):
    run_id: str
    event_type: str
    message_id: str | None = None
    document_id: str | None = None
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class ClaimEventPayload(BaseModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    reason: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    rewrite_needed: bool = False
    source_span: str
    revised_claim: str | None = None


class RewriteDiffSegment(BaseModel):
    kind: Literal["equal", "insert", "delete"]
    value: str


class RewriteEventPayload(BaseModel):
    claim_id: str
    old_span: str
    new_span: str
    diff_segments: list[RewriteDiffSegment] = Field(default_factory=list)
