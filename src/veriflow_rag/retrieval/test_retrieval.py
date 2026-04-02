import weaviate
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# --- КОНФИГУРАЦИЯ ---
# Используем те же настройки, что и при загрузке, чтобы вектора "совпадали"
print("⏳ Загрузка модели для поиска...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cpu", # Для одиночного запроса CPU более чем достаточно и надежно
    trust_remote_code=True
)
Settings.embed_model = embed_model
Settings.llm = None # Нам пока не нужен генератор, только поиск

def test_hybrid_search(query_text: str):
    print(f"\n🔎 Тестовый запрос: '{query_text}'")
    
    # 1. Подключаемся к базе
    client = weaviate.connect_to_local()
    
    try:
        # 2. Загружаем существующий индекс
        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name="VeriFlowDocs" 
        )
        # Мы НЕ используем from_documents, мы используем from_vector_store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        # 3. Создаем движок поиска (Retriever) с гибридным режимом
        # alpha=0.5 означает баланс: 50% важности вектору (смысл), 50% ключевым словам (BM25)
        retriever = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5, 
            similarity_top_k=3 # Вернуть 3 самых релевантных куска
        )

        # 4. Ищем
        results = retriever.retrieve(query_text)

        # 5. Выводим результаты
        if not results:
            print("❌ Ничего не найдено.")
            return

        print(f"✅ Найдено {len(results)} документов:\n")
        for i, node in enumerate(results):
            print(f"--- Результат #{i+1} (Score: {node.score:.4f}) ---")
            print(f"📄 Файл: {node.metadata.get('file_name', 'Unknown')}")
            # Выводим первые 200 символов найденного текста
            print(f"📝 Текст: {node.get_content()[:300]}...\n")

    finally:
        client.close()

if __name__ == "__main__":
    # Задай вопрос, который ТОЧНО есть в твоих документах
    # Например, из файла "РЕКОМЕНДАЦИИ ПО ЗАПОЛНЕНИЮ..."
    test_hybrid_search("By what percentage did Official development assistance decrease in 2024, according to the report")