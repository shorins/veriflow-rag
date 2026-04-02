from __future__ import annotations

from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    synthesis_model_name: str = "qwen/qwen3.5-9b"
    synthesis_temperature: float = 0.0
    synthesis_max_tokens: int = 700
    synthesis_top_evidence_k: int = 4
    synthesis_min_confident_evidence: int = 2
    synthesis_timeout_seconds: int = 60
    synthesis_max_evidence_chars: int = 1200

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "baseline_manifest.json"

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
