from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from veriflow_rag.retrieval.pipeline import EvidenceBlock


AnswerDepth = Literal["brief", "standard", "detailed"]


class Citation(BaseModel):
    evidence_id: str
    file_name: str
    section_title: str
    support: str


class SynthesizedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_context: bool = False
    omitted_points: list[str] = Field(default_factory=list)
    answer_depth: AnswerDepth = "standard"
    model_name: str
    prompt_version: str


class RawCitation(BaseModel):
    evidence_id: str
    file_name: str = ""
    section_title: str = ""
    support: str


class RawSynthesizedAnswer(BaseModel):
    answer: str
    citations: list[RawCitation] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_context: bool = False
    omitted_points: list[str] = Field(default_factory=list)


@dataclass
class PreparedEvidence:
    evidence_id: str
    block: EvidenceBlock
    prompt_text: str


@dataclass
class SynthesisResultBundle:
    query: str
    evidence_blocks: list[PreparedEvidence]
    synthesized_answer: SynthesizedAnswer
