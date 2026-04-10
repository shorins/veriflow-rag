from __future__ import annotations

from veriflow_rag.core.config import AppConfig, get_config
from veriflow_rag.retrieval.pipeline import EvidenceBlock, build_retriever
from veriflow_rag.synthesis.models import PreparedEvidence


class ClaimRetrievalService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever = build_retriever(use_legacy=False)

    def retrieve_for_claim(self, claim_text: str) -> list[EvidenceBlock]:
        return self.retriever.search(claim_text)

    def prepare_claim_evidence(self, evidence_blocks: list[EvidenceBlock]) -> list[PreparedEvidence]:
        prepared: list[PreparedEvidence] = []
        for index, block in enumerate(evidence_blocks[: self.config.verification_top_claim_evidence_k], start=1):
            text = block.expanded_text.strip()
            if len(text) > self.config.verification_max_evidence_chars:
                text = text[: self.config.verification_max_evidence_chars].rstrip() + "..."
            prepared.append(
                PreparedEvidence(
                    evidence_id=f"ev_{index}",
                    block=block,
                    prompt_text=(
                        f"<evidence_item id=\"ev_{index}\">\n"
                        f"<metadata>\n"
                        f"file_name: {block.file_name}\n"
                        f"section_title: {block.section_title}\n"
                        f"page_span: {block.page_span}\n"
                        f"confidence: {block.confidence_label}\n"
                        f"</metadata>\n"
                        f"<content>\n{text}\n</content>\n"
                        f"</evidence_item>"
                    ),
                )
            )
        return prepared


def build_claim_retrieval_service() -> ClaimRetrievalService:
    return ClaimRetrievalService(get_config())
