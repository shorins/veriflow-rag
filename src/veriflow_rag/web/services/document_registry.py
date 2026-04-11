from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from veriflow_rag.core.config import AppConfig
from veriflow_rag.ingestion.ingest import ingest_documents
from veriflow_rag.web.schemas import DocumentRecord


DOCUMENT_LIST_ADAPTER = TypeAdapter(list[DocumentRecord])


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.registry_path = config.documents_registry_path
        self._items: dict[str, DocumentRecord] = {}
        self._load()
        self.sync_with_data_dir()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self._items = {}
            return
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("status") == "parsed":
                    item["status"] = "uploaded"
        records = DOCUMENT_LIST_ADAPTER.validate_python(raw)
        self._items = {item.document_id: item for item in records}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = DOCUMENT_LIST_ADAPTER.dump_python(
            sorted(self._items.values(), key=lambda item: item.uploaded_at),
            mode="json",
        )
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def sync_with_data_dir(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        known_paths = {Path(item.stored_path).resolve(): item.document_id for item in self._items.values()}
        actual_paths = {path.resolve() for path in self.config.data_dir.glob("*.pdf")}

        removed_ids = [doc_id for path, doc_id in known_paths.items() if path not in actual_paths]
        for doc_id in removed_ids:
            self._items.pop(doc_id, None)

        for path in sorted(actual_paths):
            if path in known_paths:
                continue
            stat = path.stat()
            document_id = uuid4().hex
            self._items[document_id] = DocumentRecord(
                document_id=document_id,
                file_name=path.name,
                stored_path=str(path),
                size_bytes=stat.st_size,
                uploaded_at=utc_now(),
                status="uploaded",
            )
        self._save()

    def list_documents(self) -> list[DocumentRecord]:
        self.sync_with_data_dir()
        return sorted(self._items.values(), key=lambda item: item.uploaded_at, reverse=True)

    def get(self, document_id: str) -> DocumentRecord:
        item = self._items.get(document_id)
        if item is None:
            raise KeyError(document_id)
        return item

    def _unique_target_path(self, file_name: str) -> Path:
        target = self.config.data_dir / file_name
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        counter = 2
        while True:
            candidate = self.config.data_dir / f"{stem} ({counter}){suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def upload(self, source_path: Path, original_name: str | None = None) -> DocumentRecord:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        file_name = original_name or source_path.name
        target = self._unique_target_path(file_name)
        shutil.copy2(source_path, target)
        stat = target.stat()
        record = DocumentRecord(
            document_id=uuid4().hex,
            file_name=target.name,
            stored_path=str(target),
            size_bytes=stat.st_size,
            uploaded_at=utc_now(),
            status="uploaded",
        )
        self._items[record.document_id] = record
        for document_id, item in list(self._items.items()):
            if document_id == record.document_id:
                continue
            if item.status == "indexed":
                self._items[document_id] = item.model_copy(update={"status": "stale"})
        self._save()
        return record

    def _update(self, document_id: str, **changes) -> DocumentRecord:
        record = self.get(document_id)
        updated = record.model_copy(update=changes)
        self._items[document_id] = updated
        self._save()
        return updated

    def mark_error(self, document_id: str, error_message: str) -> DocumentRecord:
        return self._update(document_id, status="error", error_message=error_message)

    def _mark_all_indexed(self) -> None:
        timestamp = utc_now()
        for document_id, record in list(self._items.items()):
            if not Path(record.stored_path).exists():
                continue
            self._items[document_id] = record.model_copy(
                update={
                    "status": "indexed",
                    "last_parsed_at": timestamp,
                    "last_indexed_at": timestamp,
                    "error_message": None,
                }
            )
        self._save()

    def _mark_indexed_docs_stale(self) -> None:
        changed = False
        for document_id, record in list(self._items.items()):
            if record.status == "indexed":
                self._items[document_id] = record.model_copy(update={"status": "stale"})
                changed = True
        if changed:
            self._save()

    def reindex_corpus(self) -> list[DocumentRecord]:
        try:
            ingest_documents(str(self.config.data_dir))
        except Exception as exc:
            for document_id in list(self._items):
                self.mark_error(document_id, str(exc))
            raise
        self._mark_all_indexed()
        return self.list_documents()

    def delete(self, document_id: str) -> None:
        record = self.get(document_id)
        path = Path(record.stored_path)
        if path.exists():
            path.unlink()
        self._items.pop(document_id, None)
        self._mark_indexed_docs_stale()
        self._save()
