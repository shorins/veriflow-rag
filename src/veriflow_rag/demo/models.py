from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from veriflow_rag.core.config import DemoFaultMode


DemoFaultType = Literal["overgeneralization", "list_swap", "category_mix", "unsupported_detail"]


class InjectedFaultSpan(BaseModel):
    claim_id: str
    fault_type: DemoFaultType
    original_span: str
    injected_span: str
    source_sentence_index: int


class FaultInjectionResult(BaseModel):
    active: bool = False
    mode: DemoFaultMode = "off"
    count: int = 0
    summary: str | None = None
    answer: str | None = None
    spans: list[InjectedFaultSpan] = Field(default_factory=list)


class RawInjectedFaultSpan(BaseModel):
    fault_type: DemoFaultType
    source_sentence_index: int
    original_span: str
    injected_span: str


class RawFaultInjectionPlan(BaseModel):
    faults: list[RawInjectedFaultSpan] = Field(default_factory=list)
