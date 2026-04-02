# Типизация состояния графа (State)

from operator import add
from typing import Annotated, List, Optional, TypedDict

class TrustRagState(TypedDict):
    question: str                # Исходный вопрос пользователя
    plan: Optional[List[str]]    # План поиска (от Планировщика)
    documents: Annotated[List[str], add]         # Найденные куски текста (Context)
    draft_answer: str            # Черновой ответ (от Генератора)
    claims: Annotated[List[str], add]            # Список утверждений (от Декомпозитора)
    
    # Результаты проверки: Словарь { "Утверждение": "Статус (Entailment/Contradiction)" }
    verification_results: dict   
    
    final_answer: str            # Итоговый, исправленный ответ
    retry_count: int             # Чтобы не уйти в бесконечный цикл исправлений
