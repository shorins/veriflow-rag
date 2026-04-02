# ToDo

## Текущее состояние

- Retrieval-first baseline собран и работает локально.
- Constrained answer synthesis через `LM Studio` и `qwen/qwen3.5-9b` работает end-to-end.
- Для `definition` и части `factoid` вопросов synthesis уже дает grounded ответы с citations.
- Для вопросов вне корпуса система умеет честно отвечать `insufficient_context=true`.

## Что улучшать дальше

### 1. Улучшить retrieval для list/enumeration кейсов
- Усилить retrieval на запросах типа `Какие ...`, `Перечислите ...`, `Основные процессы ...`.
- Проверить, почему на list-like вопросах retrieval приносит слишком общие секции вместо точных подпунктов.
- Добавить более сильный учет `block_type=list/table` при ranking.
- Доработать retrieval на кейсе `Какие основные процессы жизненного цикла перечислены в ISO/IEC 12207?`.

### 2. Доработать synthesis для перечислений
- Для `enumeration` запросов требовать list-shaped answer, а не общий пересказ.
- Если evidence не содержит самих элементов списка, synthesis должен отказываться, а не отвечать общими фразами.
- Сделать prompt policy отдельно чувствительной к `definition` и `enumeration` типам вопросов.

### 3. Доработать readiness gate
- Текущая логика уже нормальна для `definition/factoid`.
- Для `enumeration/comparison/procedure` добавить более умную проверку достаточности evidence.
- Использовать не только confidence, но и форму evidence: `list`, `table`, `procedure`.

### 4. Улучшить benchmark
- Добавить больше тестовых вопросов: минимум `20-50`.
- Расширить типы вопросов:
  - определение
  - перечисление
  - факт
  - сравнение
  - процедура
- Сохранять и анализировать не только итоговую оценку, но и тип ошибки:
  - retrieval miss
  - safe refusal
  - vague synthesis
  - wrong citation

### 5. Подготовить следующий слой: verification loop
- После stabilization synthesis перейти к claim-based verification.
- Добавить разбиение ответа на утверждения.
- Для каждого утверждения искать supporting evidence.
- Помечать утверждения как:
  - supported
  - partially supported
  - unsupported
  - contradicted
- После этого добавлять repair/rewrite шаг.

## Практические технические задачи

- Оставить `just` как основной интерфейс запуска проекта.
- Следить, чтобы retrieval-модели `BAAI/bge-m3` и `BAAI/bge-reranker-v2-m3` были закешированы локально.
- При необходимости сделать отдельный `quick benchmark` на 3-5 вопросов для быстрых smoke-check прогонов.
- При необходимости добавить timeout на один benchmark-case, чтобы длинный локальный LLM run не стопорил весь прогон.

## Ссылки на текущее состояние

- Retrieval benchmark: [reports/retrieval_benchmark.md](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/reports/retrieval_benchmark.md)
- Synthesis benchmark: [reports/synthesis_benchmark.md](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/reports/synthesis_benchmark.md)
- Процесс разработки: [knowledge/Описание процесса выполнения для текста.md](/Users/sergeyshorin/Documents/Универ/ДИПЛОМ/veriflow-rag/knowledge/Описание%20процесса%20выполнения%20для%20текста.md)
