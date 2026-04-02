from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pymupdf4llm
import weaviate
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import MetadataMode, NodeRelationship
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from veriflow_rag.core.config import AppConfig, get_config


HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")


@dataclass
class ChunkManifestRecord:
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


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def infer_block_type(text: str, section_title: str, heading_path: str) -> str:
    normalized = text.lower()
    normalized_heading = f"{section_title} {heading_path}".lower()
    early_text = normalized[:400]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    bullet_lines = [
        line for line in lines
        if line.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]\s+", line)
    ]
    has_table = text.count("|") >= 4 or "\t" in text

    definition_markers = [
        "представляет собой",
        "определяется как",
        "понимается как",
        "называется",
        "рассматривается как",
        "это ",
        "is defined as",
    ]
    comparison_markers = [
        "в отличие",
        "отличается от",
        "сравнение",
        "различие",
        "versus",
        "vs",
    ]
    procedure_markers = [
        "этап",
        "шаг",
        "алгоритм",
        "порядок",
        "workflow",
        "pipeline",
        "последовательность действий",
    ]
    norm_markers = [
        "статья",
        "пункт",
        "часть",
        "кодекс",
        "закон",
        "норм",
        "регламент",
    ]
    guideline_markers = [
        "рекомендац",
        "следует",
        "необходимо",
        "показан",
        "guideline",
        "симптом",
        "лечение",
        "диагност",
    ]

    if has_table:
        return "table"
    if any(marker in normalized_heading or marker in normalized_heading for marker in comparison_markers):
        return "comparison"
    if any(marker in normalized_heading or marker in early_text for marker in definition_markers):
        return "definition"
    if any(marker in normalized_heading or marker in normalized_heading for marker in norm_markers):
        return "norm"
    if any(marker in normalized_heading or marker in normalized_heading for marker in guideline_markers):
        return "guideline"
    if any(marker in normalized_heading or marker in early_text for marker in procedure_markers):
        return "procedure"
    if len(bullet_lines) >= 3:
        return "list"
    return "paragraph"


def build_embed_model(config: AppConfig) -> HuggingFaceEmbedding:
    preferred_devices = [config.embed_device]
    if config.embed_device != "cpu":
        preferred_devices.append("cpu")

    last_error: Exception | None = None
    for device in preferred_devices:
        try:
            return HuggingFaceEmbedding(
                model_name=config.embed_model_name,
                device=device,
                trust_remote_code=True,
                embed_batch_size=1,
            )
        except Exception as exc:  # pragma: no cover - hardware specific
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not initialize embedding model.")


def configure_llama_index(config: AppConfig) -> None:
    Settings.embed_model = build_embed_model(config)
    Settings.llm = None
    Settings.chunk_size = config.child_chunk_size
    Settings.chunk_overlap = config.child_chunk_overlap


def parse_pdf_with_docling(file_path: Path) -> Document:
    reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
    docs = list(reader.lazy_load_data(file_path))
    if not docs:
        raise ValueError(f"Docling returned no documents for {file_path}")

    doc = docs[0]
    doc.metadata = {
        "file_name": file_path.name,
        "source_path": str(file_path),
        "parser_name": "docling",
    }
    return doc


def parse_pdf_with_pymupdf(file_path: Path) -> Document:
    markdown_text = pymupdf4llm.to_markdown(str(file_path))
    return Document(
        text=markdown_text,
        metadata={
            "file_name": file_path.name,
            "source_path": str(file_path),
            "parser_name": "pymupdf4llm",
        },
    )


def load_pdf_document(file_path: Path, config: AppConfig) -> Document:
    try:
        print(f"📄 Docling parsing: {file_path.name}")
        return parse_pdf_with_docling(file_path)
    except Exception as exc:
        if not config.use_docling_fallback:
            raise
        print(f"⚠️ Docling failed for {file_path.name}, falling back to pymupdf4llm: {exc}")
        return parse_pdf_with_pymupdf(file_path)


def split_markdown_into_sections(document: Document) -> list[Document]:
    text = normalize_text(document.text)
    lines = text.splitlines()

    doc_title = ""
    for line in lines:
        clean = line.strip().strip("#").strip()
        if clean:
            doc_title = clean
            break
    if not doc_title:
        doc_title = document.metadata.get("file_name", "Untitled document")

    sections: list[Document] = []
    current_heading = doc_title
    previous_heading: str | None = None
    current_lines: list[str] = []
    heading_stack: list[str] = [doc_title]

    def flush_section() -> None:
        nonlocal current_lines, previous_heading
        body = normalize_text("\n".join(current_lines))
        if not body:
            current_lines = []
            return

        heading_path = " > ".join(heading_stack)
        block_type = infer_block_type(
            text=body,
            section_title=current_heading,
            heading_path=heading_path,
        )

        sections.append(
            Document(
                text=body,
                metadata={
                    **document.metadata,
                    "doc_title": doc_title,
                    "section_title": current_heading,
                    "heading_path": heading_path,
                    "prev_heading": previous_heading,
                    "block_type": block_type,
                    "page_span": "unknown",
                },
            )
        )
        current_lines = []

    for line in lines:
        heading_match = HEADING_RE.match(line.strip())
        if heading_match and len(heading_match.group(1)) <= 2:
            flush_section()
            previous_heading = current_heading
            current_heading = heading_match.group("title").strip()
            level = len(heading_match.group(1))
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(current_heading)
            current_lines.append(line)
            continue
        current_lines.append(line)

    flush_section()
    return sections or [document]


def contextualize_child_text(metadata: dict[str, str], raw_text: str) -> str:
    prefix = [
        f"Document: {metadata.get('doc_title', metadata.get('file_name', 'Unknown document'))}",
        f"Heading path: {metadata.get('heading_path', metadata.get('section_title', 'Unknown heading'))}",
        f"Section: {metadata.get('section_title', 'Unknown section')}",
        f"Block type: {metadata.get('block_type', 'paragraph')}",
    ]
    prev_heading = metadata.get("prev_heading")
    if prev_heading:
        prefix.append(f"Previous heading: {prev_heading}")
    prefix.append("Content:")
    prefix.append(raw_text)
    return "\n".join(prefix)


def _parent_id_for_node(node) -> str:
    relationship = node.relationships.get(NodeRelationship.PARENT)
    if relationship is None:
        return ""
    return getattr(relationship, "node_id", "") or ""


def build_baseline_nodes(
    section_documents: Iterable[Document],
    config: AppConfig,
) -> tuple[list, list[ChunkManifestRecord]]:
    hierarchical_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[config.parent_chunk_size, config.child_chunk_size],
        chunk_overlap=config.child_chunk_overlap,
    )
    manifest_records: list[ChunkManifestRecord] = []
    indexed_nodes = []

    for section_doc in section_documents:
        nodes = hierarchical_parser.get_nodes_from_documents([section_doc])
        root_nodes = {node.node_id: node for node in get_root_nodes(nodes)}
        leaf_nodes = get_leaf_nodes(nodes)
        grouped_children: dict[str, list] = defaultdict(list)

        for child in leaf_nodes:
            grouped_children[_parent_id_for_node(child)].append(child)

        for parent_id, children in grouped_children.items():
            parent_node = root_nodes.get(parent_id)
            parent_text = normalize_text(
                parent_node.get_content(metadata_mode=MetadataMode.NONE) if parent_node else ""
            )
            child_count = len(children)

            for child_position, child in enumerate(children):
                raw_text = normalize_text(child.get_content(metadata_mode=MetadataMode.NONE))
                child.metadata.update(section_doc.metadata)
                child.metadata.update(
                    {
                        "chunk_level": "child",
                        "chunk_id": child.node_id,
                        "parent_id": parent_id,
                        "child_position": child_position,
                        "child_count": child_count,
                        "raw_text": raw_text,
                    }
                )
                child.set_content(contextualize_child_text(child.metadata, raw_text))
                indexed_nodes.append(child)
                manifest_records.append(
                    ChunkManifestRecord(
                        chunk_id=child.node_id,
                        parent_id=parent_id,
                        file_name=section_doc.metadata["file_name"],
                        source_path=section_doc.metadata["source_path"],
                        parser_name=section_doc.metadata["parser_name"],
                        doc_title=section_doc.metadata["doc_title"],
                        section_title=section_doc.metadata["section_title"],
                        heading_path=section_doc.metadata["heading_path"],
                        prev_heading=section_doc.metadata.get("prev_heading"),
                        block_type=section_doc.metadata["block_type"],
                        page_span=section_doc.metadata.get("page_span", "unknown"),
                        raw_text=raw_text,
                        parent_text=parent_text,
                        child_position=child_position,
                        child_count=child_count,
                    )
                )

    return indexed_nodes, manifest_records


def build_legacy_documents(file_paths: Iterable[Path]) -> list[Document]:
    docs = []
    for file_path in file_paths:
        docs.append(parse_pdf_with_pymupdf(file_path))
    return docs


def _recreate_collection(client, index_name: str) -> None:
    try:
        if client.collections.exists(index_name):
            client.collections.delete(index_name)
    except Exception:
        # If the collection API changes across versions, the index constructor
        # will create the collection when missing; we tolerate deletion failure.
        pass


def ingest_documents(data_dir: str | None = None) -> None:
    config = get_config()
    configure_llama_index(config)

    pdf_dir = Path(data_dir) if data_dir else config.data_dir
    file_paths = sorted(pdf_dir.glob("*.pdf"))
    if not file_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    all_section_docs: list[Document] = []
    for file_path in file_paths:
        pdf_doc = load_pdf_document(file_path, config)
        all_section_docs.extend(split_markdown_into_sections(pdf_doc))

    baseline_nodes, manifest_records = build_baseline_nodes(all_section_docs, config)
    config.manifest_path.write_text(
        json.dumps([asdict(record) for record in manifest_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"🧠 Saved retrieval manifest to {config.manifest_path}")

    print("🔌 Connecting to Weaviate...")
    client = weaviate.connect_to_local()
    try:
        _recreate_collection(client, config.weaviate_index_name)
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=config.weaviate_index_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(f"🚀 Indexing {len(baseline_nodes)} baseline child chunks...")
        VectorStoreIndex(
            nodes=baseline_nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        if config.use_legacy_baseline:
            _recreate_collection(client, config.legacy_weaviate_index_name)
            legacy_docs = build_legacy_documents(file_paths)
            legacy_vector_store = WeaviateVectorStore(
                weaviate_client=client,
                index_name=config.legacy_weaviate_index_name,
            )
            legacy_storage_context = StorageContext.from_defaults(vector_store=legacy_vector_store)
            print(f"🪵 Indexing legacy baseline over {len(legacy_docs)} PDFs...")
            VectorStoreIndex.from_documents(
                legacy_docs,
                storage_context=legacy_storage_context,
                transformations=[MarkdownNodeParser()],
                show_progress=True,
            )

        print("✅ Ingestion completed for baseline and legacy indexes.")
    finally:
        client.close()
        print("🔌 Weaviate connection closed.")


if __name__ == "__main__":
    ingest_documents()
