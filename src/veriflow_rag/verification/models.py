from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClaimStatus = Literal["supported", "partial", "unsupported", "contradicted"]


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    source_span: str
    source_sentence_index: int


class RawClaimList(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class VerificationEvidence(BaseModel):
    evidence_id: str
    file_name: str
    section_title: str
    support: str
    confidence_label: str


class ClaimVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    source_span: str
    source_sentence_index: int
    status: ClaimStatus
    reason: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    rewrite_needed: bool = False
    revised_claim: str | None = None
    available_evidence: list[VerificationEvidence] = Field(default_factory=list)
    residual_issue: str | None = None


class RawClaimVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    reason: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    rewrite_needed: bool = False
    revised_claim: str | None = None


class ClaimRewriteResult(BaseModel):
    claim_id: str
    rewritten_span: str


class RawClaimRewriteResult(BaseModel):
    claim_id: str
    rewritten_span: str


class AppliedRewrite(BaseModel):
    claim_id: str
    old_span: str
    new_span: str
    status_before: ClaimStatus


class VerificationRunResult(BaseModel):
    draft_answer: str
    claims: list[Claim] = Field(default_factory=list)
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list)
    highlighted_answer_html: str
    final_answer: str
    applied_rewrites: list[AppliedRewrite] = Field(default_factory=list)
    rewrite_triggered: bool = False
