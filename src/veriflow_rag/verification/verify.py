from __future__ import annotations

import json
import re

from veriflow_rag.core.config import AppConfig, get_config
from veriflow_rag.synthesis.client import LMStudioChatClient
from veriflow_rag.synthesis.models import PreparedEvidence
from veriflow_rag.verification.llm import StructuredLLMRunner
from veriflow_rag.verification.models import (
    Claim,
    ClaimVerificationResult,
    RawClaimVerificationResult,
    VerificationEvidence,
)
from veriflow_rag.verification.prompting import load_prompt_artifacts, render_user_prompt
from veriflow_rag.verification.rewrite import select_verification_profile


class ClaimVerifier:
    def __init__(self, config: AppConfig, client: LMStudioChatClient | None = None) -> None:
        self.config = config
        self.llm = StructuredLLMRunner(config, client=client)
        self.prompt_artifacts = load_prompt_artifacts("claim_verification")

    def verify_claim(self, claim: Claim, evidence_blocks: list[PreparedEvidence]) -> ClaimVerificationResult:
        profile = select_verification_profile(self.config)
        evidence_xml = "\n\n".join(item.prompt_text for item in evidence_blocks)
        schema_json = json.dumps(self.prompt_artifacts.output_schema, ensure_ascii=False, indent=2)
        user_prompt = render_user_prompt(
            self.prompt_artifacts.user_template,
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            source_span=claim.source_span,
            verification_sensitivity=profile.sensitivity,
            sensitivity_note=profile.sensitivity_note,
            evidence=evidence_xml,
            output_schema=schema_json,
        )
        raw = self.llm.run(
            model_name=getattr(self.config, "verification_model_name", self.config.synthesis_model_name),
            system_prompt=self.prompt_artifacts.system_prompt,
            user_prompt=user_prompt,
            output_schema=self.prompt_artifacts.output_schema,
            temperature=self.config.verification_temperature,
            max_tokens=self.config.verification_max_tokens,
            timeout_seconds=self.config.verification_timeout_seconds,
            response_model=RawClaimVerificationResult,
        )

        available_evidence = [
            VerificationEvidence(
                evidence_id=item.evidence_id,
                file_name=item.block.file_name,
                section_title=item.block.section_title,
                support=item.block.expanded_text[:240].strip(),
                confidence_label=item.block.confidence_label,
            )
            for item in evidence_blocks
        ]
        result = ClaimVerificationResult(
            claim_id=raw.claim_id or claim.claim_id,
            claim_text=raw.claim_text.strip() or claim.claim_text,
            source_span=claim.source_span,
            rewrite_source_span=claim.source_span,
            source_sentence_index=claim.source_sentence_index,
            status=raw.status,
            reason=raw.reason.strip(),
            used_evidence_ids=raw.used_evidence_ids,
            rewrite_needed=raw.rewrite_needed,
            revised_claim=raw.revised_claim.strip() if raw.revised_claim else None,
            available_evidence=available_evidence,
        )
        if profile.treat_partial_as_rewrite_candidate and result.status == "partial":
            if self._is_materially_broad_claim(result.claim_text):
                result.rewrite_needed = True
        return result

    @staticmethod
    def _is_materially_broad_claim(claim_text: str) -> bool:
        normalized = claim_text.lower()
        broad_markers = (
            "включает",
            "перечислены",
            "состоит",
            "различие",
            "отлич",
            "категор",
            "этапы",
            "процессы",
        )
        return any(marker in normalized for marker in broad_markers) or bool(re.search(r"\([^)]+\)", claim_text))


def build_claim_verifier() -> ClaimVerifier:
    return ClaimVerifier(get_config())
