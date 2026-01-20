import os
import pymupdf4llm # TODO: заменить на Docling от IBM
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

# --- КОНФИГУРАЦИЯ ---
# Ограничиваем аппетит модели.
# BGE-M3 мощная, но требует много памяти на длинных контекстах.
CHUNK_SIZE = 1024  # Размер одного куска текста (токенов)
BATCH_SIZE = 1     # Обрабатываем на CPU всё таки так как pytorch не хочет работать с MPS

# 1. Настройка модели Эмбеддингов (Локально на M1)
# BAAI/bge-m3 - это SOTA модель, которая отлично понимает русский и английский.
# Она создает "плотные" вектора (Dense) и разреженные (Sparse) для гибридного поиска.
print("⏳ Загрузка модели эмбеддингов на MPS...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cpu", # TODO: хардкодим cpu, так как pytorch не хочет работать с MPS, нужно сделать универсальным
    trust_remote_code=True,
    embed_batch_size=BATCH_SIZE
)
# Говорим LlamaIndex использовать эту модель глобально
Settings.embed_model = embed_model
Settings.llm = None # Пока отключаем LLM для генерации, нам нужно только индексировать
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = 50 # Небольшое перекрытие, чтобы не терять смысл на границах

def load_and_parse_pdf(file_path: str) -> list[Document]:
    """
    Локальный парсинг PDF в Markdown.
    Markdown лучше сохраняет структуру заголовков и таблиц, чем простой текст.
    """
    print(f"📄 Парсинг файла: {file_path}")
    
    # pymupdf4llm конвертирует PDF сразу в Markdown формат
    md_text = pymupdf4llm.to_markdown(file_path)
    
    # Создаем документ LlamaIndex. 
    # Важно добавить метаданные (имя файла), чтобы потом видеть источники.
    doc = Document(
        text=md_text,
        metadata={"file_name": os.path.basename(file_path)}
    )
    return [doc]

def ingest_documents():
    # 2. Подключение к Weaviate
    print("🔌 Подключение к Weaviate...")
    client = weaviate.connect_to_local()

    try:
        # 3. Настройка хранилища векторов
        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name="VeriFlowDocs" # Имя нашей коллекции (таблицы)
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 4. Чтение файлов из папки data
        data_dir = "data"
        all_docs = []
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"⚠️ Папка {data_dir} пуста. Положи туда PDF файлы!")
            return

        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_dir, filename)
                all_docs.extend(load_and_parse_pdf(file_path))

        if not all_docs:
            print("❌ Нет документов для загрузки.")
            return

        # 5. Индексация (Магия LlamaIndex)
        # Мы используем MarkdownNodeParser, чтобы он умно резал текст по заголовкам (#, ##)
        print("🚀 Начало индексации (Chunking + Embedding)...")
        
        # Специальный парсер для Markdown (лучше чем просто нарезка по словам)
        node_parser = MarkdownNodeParser()
        
        index = VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_context,
            transformations=[node_parser], # Используем умную нарезку
            show_progress=True
        )
        
        print("✅ Индексация завершена! Данные в Weaviate.")
        client.close()
    
    except Exception as e:
        print(f"🔥 Критическая ошибка при индексации: {e}")
    finally:
        client.close()
        print("🔌 Соединение с базой закрыто.")

if __name__ == "__main__":
    ingest_documents()