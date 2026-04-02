# VeriFlow RAG

Retrieval-first baseline для качественного `Hybrid RAG` без генератора. Проект индексирует PDF в `Weaviate`, использует локальный `Docling` для layout-aware parsing, строит иерархические чанки и затем выполняет retrieval в несколько стадий:

1. `Hybrid recall` через `Weaviate` (`vector + BM25`)
2. `Cross-encoder reranking`
3. `Context expansion` по соседним child chunks внутри родительского смыслового блока

## Что реализовано

- `Docling` как основной PDF parser
- fallback на `pymupdf4llm`
- иерархический chunking `parent -> child`
- baseline retrieval pipeline с reranker
- legacy baseline для сравнения качества
- benchmark script, который сравнивает old/new retrieval на PDF из `data/`

## Быстрый запуск

Поднять Weaviate:

```bash
docker compose up -d weaviate
```

Установить зависимости:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install
```

Запустить ingestion:

```bash
poetry run python -m veriflow_rag.ingestion.ingest
```

Протестировать retrieval вручную:

```bash
poetry run python -m veriflow_rag.retrieval.test_retrieval "Что такое информационная система?"
poetry run python -m veriflow_rag.retrieval.test_retrieval --legacy "Что такое информационная система?"
```

Запустить benchmark old vs new:

```bash
poetry run python -m veriflow_rag.retrieval.evaluate
```

Отчёты будут сохранены в:

- `reports/retrieval_benchmark.md`
- `reports/retrieval_benchmark.json`

## Конфигурация

Основные настройки находятся в [src/veriflow_rag/core/config.py](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/src/veriflow_rag/core/config.py):

- `WEAVIATE_INDEX_NAME`
- `LEGACY_WEAVIATE_INDEX_NAME`
- `EMBED_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `EMBED_DEVICE`
- `CHILD_CHUNK_SIZE`
- `CHILD_CHUNK_OVERLAP`
- `PARENT_CHUNK_SIZE`
- `HYBRID_ALPHA`
- `RECALL_TOP_K`
- `RERANK_TOP_N`
- `EXPAND_CONTEXT_WINDOW`

## Лицензия

Этот проект распространяется под лицензией **GNU AGPLv3**. Подробности в [LICENSE](./LICENSE).
