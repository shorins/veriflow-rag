from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DraftStrategy = Literal["conservative", "balanced", "demo"]
VerificationSensitivity = Literal["conservative", "balanced", "demo"]


def _detect_torch_device() -> str:
    """Choose the best available local device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VERIFLOW_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    artifact_dir: Path = Field(default_factory=lambda: Path.cwd() / ".cache" / "veriflow_rag")
    report_dir: Path = Field(default_factory=lambda: Path.cwd() / "reports")
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allowed_origins: Tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    weaviate_index_name: str = "VeriFlowDocsBaseline"
    legacy_weaviate_index_name: str = "VeriFlowDocsLegacy"

    embed_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    embed_device: str = Field(default_factory=_detect_torch_device)
    hf_local_files_only: bool = False

    child_chunk_size: int = 320
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1400

    hybrid_alpha: float = 0.35
    recall_top_k: int = 40
    rerank_top_n: int = 8
    expand_context_window: int = 1

    use_docling_fallback: bool = True
    use_legacy_baseline: bool = True

    lmstudio_base_url: str = "http://127.0.0.1:1234"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_api_mode: str = "auto"
    available_draft_models: Tuple[str, ...] = (
        "qwen2.5-vl-3b-instruct",
        "qwen2.5-vl-7b-instruct",
    )
    available_verification_models: Tuple[str, ...] = (
        "qwen2.5-vl-3b-instruct",
        "qwen2.5-vl-7b-instruct",
    )
    draft_model_name: str = "qwen2.5-vl-3b-instruct"
    verification_model_name: str = "qwen2.5-vl-7b-instruct"
    draft_strategy: DraftStrategy = "demo"
    verification_sensitivity: VerificationSensitivity = "demo"
    synthesis_temperature: float = 0.0
    synthesis_max_tokens: int = 700
    synthesis_top_evidence_k: int = 4
    synthesis_min_confident_evidence: int = 2
    synthesis_timeout_seconds: int = 180
    synthesis_max_evidence_chars: int = 1200

    verification_top_claim_evidence_k: int = 3
    verification_max_evidence_chars: int = 900
    verification_temperature: float = 0.0
    verification_max_tokens: int = 700
    verification_timeout_seconds: int = 180
    verification_partial_ratio_threshold: float = 0.2
    verification_problem_span_ratio_threshold: float = 0.15

    @property
    def synthesis_model_name(self) -> str:
        return self.draft_model_name

    def with_draft_model(self, model_name: str) -> "AppConfig":
        if model_name not in self.available_draft_models:
            raise ValueError(
                f"Unknown draft model '{model_name}'. Expected one of: {', '.join(self.available_draft_models)}"
            )
        return self.model_copy(update={"draft_model_name": model_name})

    def with_verification_model(self, model_name: str) -> "AppConfig":
        if model_name not in self.available_verification_models:
            raise ValueError(
                f"Unknown verification model '{model_name}'. Expected one of: {', '.join(self.available_verification_models)}"
            )
        return self.model_copy(update={"verification_model_name": model_name})

    def with_draft_strategy(self, strategy: DraftStrategy) -> "AppConfig":
        return self.model_copy(update={"draft_strategy": strategy})

    def with_verification_sensitivity(self, sensitivity: VerificationSensitivity) -> "AppConfig":
        return self.model_copy(update={"verification_sensitivity": sensitivity})

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "baseline_manifest.json"

    @property
    def documents_registry_path(self) -> Path:
        return self.artifact_dir / "documents.json"

    @property
    def benchmark_report_path(self) -> Path:
        return self.report_dir / "retrieval_benchmark.md"

    @property
    def benchmark_json_path(self) -> Path:
        return self.report_dir / "retrieval_benchmark.json"

    @property
    def synthesis_benchmark_report_path(self) -> Path:
        return self.report_dir / "synthesis_benchmark.md"

    @property
    def synthesis_benchmark_json_path(self) -> Path:
        return self.report_dir / "synthesis_benchmark.json"

    def ensure_runtime_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    config = AppConfig()
    config.ensure_runtime_dirs()
    return config
