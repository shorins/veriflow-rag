set shell := ["bash", "-uc"]

poetry := "POETRY_VIRTUALENVS_IN_PROJECT=true poetry"
python := poetry + " run python"

default:
    @just --list

install:
    {{poetry}} install

weaviate-up:
    docker compose up -d weaviate

weaviate-down:
    docker compose down

ingest:
    {{python}} -m veriflow_rag.ingestion.ingest

retrieval query="Что такое информационная система?":
    {{python}} -m veriflow_rag.retrieval.test_retrieval "{{query}}"

retrieval-legacy query="Что такое информационная система?":
    {{python}} -m veriflow_rag.retrieval.test_retrieval --legacy "{{query}}"

retrieval-benchmark:
    {{python}} -m veriflow_rag.retrieval.evaluate

synthesis query="Что такое информационная система?":
    {{python}} -m veriflow_rag.synthesis.test_synthesis "{{query}}"

synthesis-local query="Что такое информационная система?":
    VERIFLOW_HF_LOCAL_FILES_ONLY=true {{python}} -m veriflow_rag.synthesis.test_synthesis "{{query}}"

synthesis-benchmark:
    {{python}} -m veriflow_rag.synthesis.evaluate

synthesis-benchmark-local:
    VERIFLOW_HF_LOCAL_FILES_ONLY=true {{python}} -m veriflow_rag.synthesis.evaluate

verification query="Что такое информационная система?":
    {{python}} -m veriflow_rag.verification.test_verification "{{query}}"

verification-local query="Что такое информационная система?":
    VERIFLOW_HF_LOCAL_FILES_ONLY=true {{python}} -m veriflow_rag.verification.test_verification "{{query}}"

ui:
    {{poetry}} run chainlit run src/veriflow_rag/app.py -w

chainlit:
    {{poetry}} run chainlit run src/veriflow_rag/app.py -w

api:
    {{poetry}} run uvicorn veriflow_rag.web.app:app --host 127.0.0.1 --port 8000 --reload

ui-web:
    cd frontend && npm run dev

web-dev:
    trap 'kill 0' EXIT; {{poetry}} run uvicorn veriflow_rag.web.app:app --host 127.0.0.1 --port 8000 --reload & (cd frontend && npm run dev)

web-ui-test:
    trap 'kill 0' EXIT; VERIFLOW_WEB_UI_TEST_MODE=true {{poetry}} run uvicorn veriflow_rag.web.app:app --host 127.0.0.1 --port 8000 --reload & (cd frontend && npm run dev)

lmstudio-models:
    curl -sS http://127.0.0.1:1234/v1/models

lmstudio-chat prompt="Say ok":
    curl -sS -X POST http://127.0.0.1:1234/api/v1/chat \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"qwen/qwen3.5-9b\",\"system_prompt\":\"Answer briefly.\",\"input\":\"{{prompt}}\"}"

download-retrieval-models:
    {{python}} -c "from huggingface_hub import snapshot_download; [print(f'{repo_id}: {snapshot_download(repo_id=repo_id)}') for repo_id in ('BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3')]"

test:
    {{python}} -m unittest tests.test_retrieval_pipeline tests.test_synthesis_service tests.test_verification_pipeline
