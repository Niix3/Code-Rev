# Multi-Agent System with LangGraph

Система мультиагентов на базе LangGraph с pipeline: `architect -> coding -> write_tests -> run_tests -> review -> composer`.

## Архитектура

```
User
 ↓
API Gateway (FastAPI)
 ↓
LangGraph Orchestrator
 ├── Architect Agent
 ├── Coding Agent (OpenHands)
 ├── Tester Agent (write_tests → run_tests)
 ├── Correctness / Security / Code Style Reviewers
 └── Composer Review Agent (pass | refine)
 ↓
Response Aggregation
 ↓
User
```

## Компоненты

### 1. API Gateway (FastAPI)
- RESTful API для взаимодействия с системой
- Поддержка текстовых запросов

### 2. LangGraph Orchestrator
- Управление потоком выполнения между агентами
- Агрегация ответов

### 3. Агенты

#### Architect Agent
- Формирует JSON-план с `explanation` и `steps` (agent, description, file, …)

#### Coding Agent (OpenHands)
- Реализует код по архитектурному плану

#### Tester Agent
- **write_tests** — пишет pytest-тесты после coding
- **run_tests** — запускает `pytest -q`

#### Review pipeline
- **Correctness Reviewer** — соответствие запросу, качеству тестов и результатам прогона
- **Security Reviewer** — проверка безопасности кода
- **Code Style Reviewer** — соответствие стандартам оформления
- **Composer Review Agent** — собирает вердикт `pass` или `refine` (Pydantic); при `refine` coding agent получает структурированную критику

#### Response Aggregator
- Финальная агрегация ответов этапов pipeline

### 4. Tools

#### Web Search
- Интеграция с поисковыми API (Tavily, SerpAPI)

#### Python Executor
- Безопасное выполнение Python кода
- Guardrails на опасные операции
- Ограничения на импорты

## Установка

1. Клонируйте репозиторий или создайте проект

2. Создайте файл `.env`:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.aitunnel.ru/v1/
DEFAULT_LLM_MODEL=gpt-4-turbo-preview
WORKSPACE_PATH=/workspace
OPENHANDS_API_KEY=your_openhands_or_llm_key
OPENHANDS_MODEL=openhands/claude-sonnet-4-5-20250929
OPENHANDS_LLM_BASE_URL=
TESTER_COMMAND=pytest -q
```

3. Запустите систему в Docker:
```bash
docker compose up --build
```

## Запуск

Локально без Docker (опционально):

```bash
pip install -r requirements.txt
python main.py
```

По умолчанию API доступен по адресу: `http://localhost:8000`

Документация API: `http://localhost:8000/docs`

## Использование

### Текстовый запрос

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
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
│   ├── architect_agent.py
│   ├── coding_agent.py
│   ├── tester_agent.py
│   ├── correctness_reviewer.py
│   ├── security_reviewer.py
│   ├── code_style_reviewer.py
│   ├── composer_review_agent.py
│   ├── review_models.py
│   ├── response_aggregator.py
│   ├── tool_agent.py
├── tools/                 # Внешние инструменты
│   ├── __init__.py
│   ├── web_search.py
│   └── python_executor.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Конфигурация

Настройки находятся в `config/settings.py` и могут быть переопределены через переменные окружения в `.env`:

- `OPENAI_API_KEY`: API ключ OpenAI
- `DEFAULT_LLM_MODEL`: Модель LLM по умолчанию
- `WORKSPACE_PATH`: путь к общей рабочей директории (volume)
- `OPENHANDS_API_KEY`: ключ для OpenHands SDK (если пусто, используется `OPENAI_API_KEY`)
- `OPENHANDS_MODEL`: модель для OpenHands SDK
- `OPENHANDS_LLM_BASE_URL`: кастомный base URL для LLM провайдера (опционально)
- `TESTER_COMMAND`: команда тестирования в shared workspace

## Расширение системы

### Добавление нового агента

1. Создайте файл в `agents/` с классом агента
2. Добавьте узел в `orchestrator/graph.py`
3. Подключите узел в pipeline и экспортируйте агент в `agents/__init__.py`

### Добавление нового инструмента

1. Создайте класс инструмента в `tools/`
2. Добавьте его в `ToolAgent` в `agents/tool_agent.py`

## Лицензия

MIT

