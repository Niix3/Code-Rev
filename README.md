# Multi-Agent System with LangGraph

Архитектура системы мультиагентов на базе LangGraph с поддержкой мультимодальных запросов и RAG.

## Архитектура

```
User
 ↓
API Gateway (FastAPI)
 ↓
LangGraph Orchestrator
 ├── Router Agent
 │
 ├── Text Reasoning Agent
 ├── Vision Agent
 ├── Retrieval Agent (RAG)
 ├── Tool Agent
 │
 └── Critic / Verifier Agent
 ↓
Response Aggregation
 ↓
User
```

## Компоненты

### 1. API Gateway (FastAPI)
- RESTful API для взаимодействия с системой
- Поддержка текстовых и мультимодальных запросов
- Управление документами для RAG

### 2. LangGraph Orchestrator
- Управление потоком выполнения между агентами
- Маршрутизация запросов
- Агрегация ответов

### 3. Агенты

#### Router Agent
- Анализирует запрос и определяет подходящего агента
- Учитывает наличие изображений

#### Text Reasoning Agent
- Сложные рассуждения и анализ
- Пошаговое решение проблем

#### Vision Agent
- Анализ изображений
- OCR и captioning
- Мультимодальное понимание

#### Retrieval Agent (RAG)
- Поиск в векторной базе данных
- Переписывание запросов
- Сжатие контекста
- Переранжирование результатов

#### Tool Agent
- Выполнение внешних инструментов
- Web search
- Python execution (с guardrails)
- Интеграция с API

#### Critic/Verifier Agent
- Проверка качества ответов
- Оценка релевантности и точности
- Агрегация множественных ответов

### 4. Multimodal Pipeline

#### Image Processing
- Image encoder (CLIP/SigLIP)
- Captioning (GPT-4V)
- OCR

#### Fusion Strategies
- Early fusion (embeddings)
- Late fusion (textual grounding)
- Cross-attention (advanced)

### 5. RAG Layer

#### Vector Store
- FAISS (локально)
- Qdrant (опционально)
- Neo4j (опционально)

#### Retrieval Strategies
- Query rewriting
- Multimodal retrieval
- Context compression
- Re-ranking

### 6. Tools

#### Web Search
- Интеграция с поисковыми API (Tavily, SerpAPI)

#### Python Executor
- Безопасное выполнение Python кода
- Guardrails на опасные операции
- Ограничения на импорты

## Установка

1. Клонируйте репозиторий или создайте проект

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

4. Заполните `.env` файл с вашими API ключами:
```
OPENAI_API_KEY=your_key_here
```

5. (Опционально) Настройте векторную базу данных:
   - Для Qdrant: установите и запустите Qdrant
   - Для Neo4j: установите и запустите Neo4j

## Запуск

```bash
python main.py
```

API будет доступен по адресу: `http://localhost:8000`

Документация API: `http://localhost:8000/docs`

## Использование

### Текстовый запрос

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Мультимодальный запрос

```bash
curl -X POST "http://localhost:8000/query-multimodal" \
  -F "query=What is in this image?" \
  -F "image=@path/to/image.jpg"
```

### Добавление документов в RAG

```bash
curl -X POST "http://localhost:8000/rag/add-documents" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Document 1 text...", "Document 2 text..."]}'
```

## Структура проекта

```
.
├── main.py                 # FastAPI gateway
├── config/                 # Конфигурация
│   ├── __init__.py
│   └── settings.py
├── orchestrator/           # LangGraph orchestrator
│   ├── __init__.py
│   └── graph.py
├── agents/                 # Все агенты
│   ├── __init__.py
│   ├── router_agent.py
│   ├── text_reasoning_agent.py
│   ├── vision_agent.py
│   ├── retrieval_agent.py
│   ├── tool_agent.py
│   └── critic_agent.py
├── multimodal/            # Мультимодальная обработка
│   ├── __init__.py
│   └── image_encoder.py
├── rag/                   # RAG компоненты
│   ├── __init__.py
│   ├── vector_store.py
│   └── query_processor.py
├── tools/                 # Внешние инструменты
│   ├── __init__.py
│   ├── web_search.py
│   └── python_executor.py
├── requirements.txt
└── README.md
```

## Конфигурация

Настройки находятся в `config/settings.py` и могут быть переопределены через переменные окружения в `.env`:

- `OPENAI_API_KEY`: API ключ OpenAI
- `DEFAULT_LLM_MODEL`: Модель LLM по умолчанию
- `VISION_MODEL`: Модель для vision задач
- `EMBEDDING_MODEL`: Модель для embeddings
- `ENABLE_WEB_SEARCH`: Включить web search
- `ENABLE_PYTHON_EXEC`: Включить Python execution
- `MAX_TOOL_CALLS`: Максимальное количество вызовов инструментов

## Расширение системы

### Добавление нового агента

1. Создайте файл в `agents/` с классом агента
2. Добавьте узел в `orchestrator/graph.py`
3. Обновите router для маршрутизации к новому агенту

### Добавление нового инструмента

1. Создайте класс инструмента в `tools/`
2. Добавьте его в `ToolAgent` в `agents/tool_agent.py`

### Настройка RAG

1. Инициализируйте векторную базу данных
2. Добавьте документы через API или напрямую
3. Настройте параметры retrieval в `RetrievalAgent`

## Лицензия

MIT

