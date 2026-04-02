# VeriFlow RAG

Retrieval-first baseline для качественного `Hybrid RAG` и первый слой constrained answer synthesis. Проект индексирует PDF в `Weaviate`, использует локальный `Docling` для layout-aware parsing, строит иерархические чанки и затем выполняет retrieval в несколько стадий:

1. `Hybrid recall` через `Weaviate` (`vector + BM25`)
2. `Cross-encoder reranking`
3. `Context expansion` по соседним child chunks внутри родительского смыслового блока
4. `Constrained answer synthesis` через локальную LLM в `LM Studio`

## Что реализовано

- `Docling` как основной PDF parser
- fallback на `pymupdf4llm`
- иерархический chunking `parent -> child`
- baseline retrieval pipeline с reranker
- legacy baseline для сравнения качества
- benchmark script, который сравнивает old/new retrieval на PDF из `data/`
- synthesis layer поверх `EvidenceBlock[]`
- prompts и JSON schema как versioned artifacts в репозитории
- synthesis benchmark для grounded ответов с citations

## Быстрый запуск

Основной способ работы с проектом теперь через `just`.

Установить зависимости проекта:

```bash
just install
```

Поднять Weaviate:

```bash
just weaviate-up
```

Проиндексировать PDF:

```bash
just ingest
```

Проверить retrieval:

```bash
just retrieval "Что такое информационная система?"
just retrieval-legacy "Что такое информационная система?"
```

Запустить retrieval benchmark:

```bash
just retrieval-benchmark
```

Проверить доступность LM Studio:

```bash
just lmstudio-models
just lmstudio-chat "Say ok"
```

Запустить constrained answer synthesis:

```bash
just synthesis "Что такое информационная система?"
```

Если retrieval-модели уже лежат локально и нужен полностью офлайн-прогон:

```bash
just synthesis-local "Что такое информационная система?"
```

Запустить synthesis benchmark:

```bash
just synthesis-benchmark
just synthesis-benchmark-local
```

Отчёты будут сохранены в:

- `reports/retrieval_benchmark.md`
- `reports/retrieval_benchmark.json`
- `reports/synthesis_benchmark.md`
- `reports/synthesis_benchmark.json`

## Конфигурация

Основные настройки находятся в [src/veriflow_rag/core/config.py](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/src/veriflow_rag/core/config.py):

- `WEAVIATE_INDEX_NAME`
- `LEGACY_WEAVIATE_INDEX_NAME`
- `EMBED_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `EMBED_DEVICE`
- `HF_LOCAL_FILES_ONLY`
- `CHILD_CHUNK_SIZE`
- `CHILD_CHUNK_OVERLAP`
- `PARENT_CHUNK_SIZE`
- `HYBRID_ALPHA`
- `RECALL_TOP_K`
- `RERANK_TOP_N`
- `EXPAND_CONTEXT_WINDOW`
- `LMSTUDIO_BASE_URL`
- `LMSTUDIO_API_KEY`
- `SYNTHESIS_MODEL_NAME`
- `SYNTHESIS_TEMPERATURE`
- `SYNTHESIS_MAX_TOKENS`
- `SYNTHESIS_TOP_EVIDENCE_K`
- `SYNTHESIS_MIN_CONFIDENT_EVIDENCE`
- `SYNTHESIS_TIMEOUT_SECONDS`

## Just команды

Ключевые команды из [justfile](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/justfile):

- `just install`
- `just weaviate-up`
- `just ingest`
- `just retrieval "..."` / `just retrieval-legacy "..."`
- `just retrieval-benchmark`
- `just synthesis "..."`
- `just synthesis-local "..."`
- `just synthesis-benchmark`
- `just synthesis-benchmark-local`
- `just lmstudio-models`
- `just lmstudio-chat "..."`
- `just download-retrieval-models`
- `just test`

## Synthesis слой

`Constrained answer synthesis` работает так:

1. retrieval возвращает `EvidenceBlock[]`
2. сервис выбирает top `3-5` блоков и присваивает им `evidence_id`
3. в локальную LLM передаются только эти evidence blocks
4. модель обязана вернуть строго JSON:
   - `answer`
   - `citations`
   - `used_evidence_ids`
   - `insufficient_context`
   - `omitted_points`
5. если retrieval слабый или JSON невалидный, система честно возвращает `insufficient_context=true`

Рекомендуемый локальный synthesis model baseline: `Qwen3.5-9B` в `LM Studio`.

Если retrieval-модели уже закешированы локально и нужен полностью локальный запуск без обращений к Hugging Face, включите:

```bash
just download-retrieval-models
just synthesis-local "Что такое информационная система?"
```

В этом режиме retrieval будет брать `BAAI/bge-m3` и `BAAI/bge-reranker-v2-m3` только из локального кэша.
Если одной из моделей нет в `~/.cache/huggingface/hub`, система вернет явную ошибку о недостающем локальном snapshot.

## Лицензия

Этот проект распространяется под лицензией **GNU AGPLv3**. Подробности в [LICENSE](./LICENSE).
