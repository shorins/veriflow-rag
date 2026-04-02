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

    child_chunk_size: int = 320
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1400

    hybrid_alpha: float = 0.35
    recall_top_k: int = 40
    rerank_top_n: int = 8
    expand_context_window: int = 1

    use_docling_fallback: bool = True
    use_legacy_baseline: bool = True

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "baseline_manifest.json"

    @property
    def benchmark_report_path(self) -> Path:
        return self.report_dir / "retrieval_benchmark.md"

    @property
    def benchmark_json_path(self) -> Path:
        return self.report_dir / "retrieval_benchmark.json"

    def ensure_runtime_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    config = AppConfig()
    config.ensure_runtime_dirs()
    return config
