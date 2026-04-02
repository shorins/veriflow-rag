from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import weaviate
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from sentence_transformers import CrossEncoder

from veriflow_rag.core.config import AppConfig, get_config


STOPWORDS = {
    "и", "в", "во", "на", "по", "для", "из", "с", "со", "к", "что", "как", "какие",
    "какой", "какая", "какое", "это", "при", "ли", "над", "под", "об", "о", "от",
    "the", "a", "an", "of", "to", "is", "are",
}

COMMON_SUFFIXES = [
    "ирования", "ированиям", "ированиях", "ениями", "ениями", "остью", "ности", "ционный",
    "ционная", "ционные", "ционного", "ционном", "альными", "ального", "альному",
    "иями", "иями", "иями", "иями", "ами", "ями", "ого", "ему", "ыми", "ими",
    "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ий", "ый", "ому", "ах", "ях",
    "ам", "ям", "ов", "ев", "ом", "ем", "а", "я", "ы", "и", "е", "у", "ю",
]


@dataclass
class QueryProfile:
    intent: str
    domain: str
    has_numeric_anchor: bool


@dataclass
class ChunkRecord:
    chunk_id: str
    parent_id: str
    file_name: str
    source_path: str
    parser_name: str
    doc_title: str
    section_title: str
    heading_path: str
    prev_heading: str | None
    block_type: str
    page_span: str
    raw_text: str
    parent_text: str
    child_position: int
    child_count: int


@dataclass
class EvidenceBlock:
    query: str
    rank: int
    retrieval_score: float | None
    rerank_score: float | None
    file_name: str
    page_span: str
    section_title: str
    text: str
    parent_text: str
    expanded_text: str
    confidence_label: str


class SentenceCrossEncoderReranker:
    def __init__(self, model_name: str, device: str, local_files_only: bool = False) -> None:
        preferred_devices = [device]
        if device != "cpu":
            preferred_devices.append("cpu")

        last_error: Exception | None = None
        for candidate in preferred_devices:
            try:
                self.model = CrossEncoder(
                    model_name,
                    device=candidate,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                return
            except Exception as exc:  # pragma: no cover - hardware specific
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not initialize reranker.")

    def rerank(self, query: str, nodes: Iterable[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        nodes = list(nodes)
        if not nodes:
            return []

        pairs = [
            (query, node.node.metadata.get("raw_text") or node.node.get_content(metadata_mode=MetadataMode.NONE))
            for node in nodes
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for node, score in zip(nodes, scores):
            node.score = float(score)

        return sorted(nodes, key=lambda item: item.score or float("-inf"), reverse=True)[:top_n]


class RetrieverService:
    def __init__(self, config: AppConfig, use_legacy: bool = False) -> None:
        self.config = config
        self.use_legacy = use_legacy
        self.manifest = self._load_manifest(config.manifest_path)
        self.records_by_id = {record.chunk_id: record for record in self.manifest}
        self.records_by_parent: dict[str, list[ChunkRecord]] = defaultdict(list)
        for record in self.manifest:
            self.records_by_parent[record.parent_id].append(record)

        for records in self.records_by_parent.values():
            records.sort(key=lambda item: item.child_position)

        self.reranker = None if use_legacy else self._build_reranker()
        self._configure_llama_index()

    def _build_reranker(self) -> SentenceCrossEncoderReranker:
        model_name = self._resolve_model_source(self.config.reranker_model_name)
        with self._huggingface_mode():
            return SentenceCrossEncoderReranker(
                model_name=model_name,
                device=self.config.embed_device,
                local_files_only=self.config.hf_local_files_only,
            )

    @contextmanager
    def _huggingface_mode(self):
        if not self.config.hf_local_files_only:
            yield
            return

        previous = {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        }
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _load_manifest(path: Path) -> list[ChunkRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ChunkRecord(**item) for item in payload]

    def _resolve_model_source(self, model_name: str) -> str:
        if not self.config.hf_local_files_only:
            return model_name

        if Path(model_name).exists():
            return str(Path(model_name))

        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            raise RuntimeError(
                f"Local-only mode is enabled, but model '{model_name}' is not cached in {snapshots_dir}."
            )

        snapshot_paths = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if not snapshot_paths:
            raise RuntimeError(
                f"Local-only mode is enabled, but no snapshots were found for model '{model_name}'."
            )
        return str(snapshot_paths[-1])

    def _configure_llama_index(self) -> None:
        preferred_devices = [self.config.embed_device]
        if self.config.embed_device != "cpu":
            preferred_devices.append("cpu")

        last_error: Exception | None = None
        for device in preferred_devices:
            try:
                model_name = self._resolve_model_source(self.config.embed_model_name)
                with self._huggingface_mode():
                    Settings.embed_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        device=device,
                        trust_remote_code=True,
                        local_files_only=self.config.hf_local_files_only,
                    )
                Settings.llm = None
                return
            except Exception as exc:  # pragma: no cover - hardware specific
                last_error = exc
        if last_error is not None:
            raise last_error
        Settings.llm = None

    def _index_name(self) -> str:
        return (
            self.config.legacy_weaviate_index_name
            if self.use_legacy
            else self.config.weaviate_index_name
        )

    def _expand_context(self, record: ChunkRecord) -> str:
        siblings = self.records_by_parent.get(record.parent_id, [record])
        start = max(0, record.child_position - self.config.expand_context_window)
        end = min(len(siblings), record.child_position + self.config.expand_context_window + 1)
        return "\n\n".join(item.raw_text for item in siblings[start:end])

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.lower()
        for suffix in COMMON_SUFFIXES:
            if len(token) > 6 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            RetrieverService._normalize_token(token)
            for token in re.findall(r"[a-zA-Zа-яА-Я0-9/]+", text.lower())
            if token not in STOPWORDS and len(token) > 1
        ]

    def _classify_query(self, query: str) -> QueryProfile:
        normalized = query.lower()

        comparison_markers = ["разница", "отлич", "сравн", "versus", "vs", "difference"]
        procedure_markers = ["как", "порядок", "этап", "шаг", "процед", "алгоритм", "workflow"]
        enumeration_markers = ["какие", "перечис", "список", "виды", "типы", "элементы", "симптомы"]
        definition_markers = ["что такое", "что представляет собой", "определение", "what is", "define"]
        definition_list_markers = ["что определяет", "что включает", "что содержит", "что предусматривает"]

        if any(marker in normalized for marker in definition_markers):
            intent = "definition"
        elif any(marker in normalized for marker in comparison_markers):
            intent = "comparison"
        elif any(marker in normalized for marker in definition_list_markers):
            intent = "enumeration"
        elif any(marker in normalized for marker in enumeration_markers):
            intent = "enumeration"
        elif any(marker in normalized for marker in procedure_markers):
            intent = "procedure"
        else:
            intent = "factoid"

        legal_markers = ["статья", "кодекс", "закон", "право", "договор", "норма", "court", "regulation"]
        medical_markers = [
            "симптом", "лечение", "диагноз", "диагности", "пациент", "терап", "синдром",
            "медици", "guideline", "dosage",
        ]

        if any(marker in normalized for marker in legal_markers):
            domain = "legal"
        elif any(marker in normalized for marker in medical_markers):
            domain = "medical"
        else:
            domain = "general"

        has_numeric_anchor = bool(re.search(r"\d", normalized))
        return QueryProfile(intent=intent, domain=domain, has_numeric_anchor=has_numeric_anchor)

    def _heading_overlap_boost(self, query: str, record: ChunkRecord) -> float:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return 0.0

        heading_text = f"{record.heading_path} {record.section_title} {record.prev_heading or ''}"
        heading_tokens = set(self._tokenize(heading_text))
        if not heading_tokens:
            return 0.0

        overlap = query_tokens & heading_tokens
        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / max(1, min(len(query_tokens), len(heading_tokens)))
        bonus = 0.2 if len(overlap) >= 2 else 0.0
        return overlap_ratio * 0.45 + bonus

    def _intent_block_boost(self, profile: QueryProfile, record: ChunkRecord) -> float:
        mapping = {
            "definition": {
                "definition": 0.9,
                "paragraph": 0.2,
                "table": 0.1,
            },
            "enumeration": {
                "list": 0.9,
                "table": 0.8,
                "procedure": 0.45,
                "guideline": 0.35,
            },
            "comparison": {
                "comparison": 0.9,
                "table": 0.65,
                "paragraph": 0.2,
            },
            "procedure": {
                "procedure": 0.9,
                "guideline": 0.7,
                "list": 0.45,
            },
            "factoid": {
                "norm": 0.55,
                "table": 0.45,
                "paragraph": 0.25,
                "definition": 0.2,
            },
        }
        return mapping.get(profile.intent, {}).get(record.block_type, 0.0)

    def _domain_block_boost(self, profile: QueryProfile, record: ChunkRecord) -> float:
        if profile.domain == "legal":
            mapping = {"norm": 0.8, "table": 0.25, "paragraph": 0.1}
            return mapping.get(record.block_type, 0.0)
        if profile.domain == "medical":
            mapping = {"guideline": 0.75, "table": 0.55, "list": 0.45, "procedure": 0.25}
            return mapping.get(record.block_type, 0.0)
        return 0.0

    def _numeric_anchor_boost(self, profile: QueryProfile, record: ChunkRecord) -> float:
        if not profile.has_numeric_anchor:
            return 0.0
        text = f"{record.heading_path} {record.raw_text}".lower()
        return 0.25 if re.search(r"\d", text) else 0.0

    def _structural_boost(self, query: str, profile: QueryProfile, record: ChunkRecord) -> float:
        return (
            self._heading_overlap_boost(query, record)
            + self._intent_block_boost(profile, record)
            + self._domain_block_boost(profile, record)
            + self._numeric_anchor_boost(profile, record)
        )

    @staticmethod
    def _confidence_label(rank: int, rerank_score: float | None) -> str:
        if rerank_score is None:
            return "baseline"
        if rank == 1 and rerank_score >= 0.6:
            return "high"
        if rerank_score >= 0.2:
            return "medium"
        return "low"

    def search(self, query: str) -> list[EvidenceBlock]:
        client = weaviate.connect_to_local()
        try:
            vector_store = WeaviateVectorStore(
                weaviate_client=client,
                index_name=self._index_name(),
            )
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            retriever = index.as_retriever(
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                alpha=self.config.hybrid_alpha,
                similarity_top_k=(3 if self.use_legacy else self.config.recall_top_k),
            )
            candidates = list(retriever.retrieve(query))
            for candidate in candidates:
                candidate.node.metadata["retrieval_score"] = candidate.score

            if self.use_legacy:
                return [
                    EvidenceBlock(
                        query=query,
                        rank=rank,
                        retrieval_score=node.score,
                        rerank_score=None,
                        file_name=node.node.metadata.get("file_name", "Unknown"),
                        page_span=node.node.metadata.get("page_span", "unknown"),
                        section_title=node.node.metadata.get("section_title", "legacy-chunk"),
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        parent_text=node.get_content(metadata_mode=MetadataMode.NONE),
                        expanded_text=node.get_content(metadata_mode=MetadataMode.NONE),
                        confidence_label="baseline",
                    )
                    for rank, node in enumerate(candidates, start=1)
                ]

            query_profile = self._classify_query(query)
            reranked = self.reranker.rerank(query, candidates, self.config.rerank_top_n)
            grouped: dict[str, dict] = {}
            for node in reranked:
                chunk_id = node.node.metadata.get("chunk_id") or node.node.node_id
                record = self.records_by_id.get(chunk_id)
                if record is None:
                    continue

                bucket = grouped.setdefault(
                    record.parent_id,
                    {
                        "record": record,
                        "retrieval_score": node.node.metadata.get("retrieval_score"),
                        "rerank_score": node.score,
                        "best_child_score": node.score,
                        "best_rank": len(grouped) + 1,
                    },
                )
                bucket["retrieval_score"] = max(
                    bucket["retrieval_score"] if bucket["retrieval_score"] is not None else float("-inf"),
                    node.node.metadata.get("retrieval_score")
                    if node.node.metadata.get("retrieval_score") is not None
                    else float("-inf"),
                )
                bucket["rerank_score"] = max(
                    bucket["rerank_score"] if bucket["rerank_score"] is not None else float("-inf"),
                    node.score if node.score is not None else float("-inf"),
                )
                bucket["structural_boost"] = max(
                    bucket.get("structural_boost", float("-inf")),
                    self._structural_boost(query, query_profile, record),
                )
                if (node.score or float("-inf")) > (bucket.get("best_child_score") or float("-inf")):
                    bucket["record"] = record
                    bucket["best_child_score"] = node.score

            ordered = sorted(
                grouped.values(),
                key=lambda item: (
                    (item["rerank_score"] if item["rerank_score"] is not None else float("-inf"))
                    + item.get("structural_boost", 0.0)
                ),
                reverse=True,
            )

            evidence_blocks: list[EvidenceBlock] = []
            for rank, item in enumerate(ordered, start=1):
                record = item["record"]
                expanded_text = self._expand_context(record)
                evidence_blocks.append(
                    EvidenceBlock(
                        query=query,
                        rank=rank,
                        retrieval_score=item["retrieval_score"],
                        rerank_score=item["rerank_score"],
                        file_name=record.file_name,
                        page_span=record.page_span,
                        section_title=record.section_title,
                        text=record.raw_text,
                        parent_text=record.parent_text,
                        expanded_text=expanded_text,
                        confidence_label=self._confidence_label(rank, item["rerank_score"]),
                    )
                )

            return evidence_blocks
        finally:
            client.close()


def build_retriever(use_legacy: bool = False) -> RetrieverService:
    return RetrieverService(get_config(), use_legacy=use_legacy)
