# Quick Start Guide

## Быстрая установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка окружения

Создайте файл `.env` в корне проекта:

```env
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_LLM_MODEL=gpt-4-turbo-preview
VISION_MODEL=gpt-4-vision-preview
EMBEDDING_MODEL=text-embedding-3-large
```

### 3. Запуск сервера

```bash
python main.py
```

Сервер запустится на `http://localhost:8000`

### 4. Проверка работоспособности

Откройте в браузере: `http://localhost:8000/docs` для интерактивной документации API

Или используйте curl:

```bash
curl http://localhost:8000/health
```

### 5. Примеры использования

#### Текстовый запрос

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?"}'
```

#### Мультимодальный запрос (с изображением)

```bash
curl -X POST "http://localhost:8000/query-multimodal" \
  -F "query=What is in this image?" \
  -F "image=@path/to/image.jpg"
```

#### Добавление документов в RAG

```bash
curl -X POST "http://localhost:8000/rag/add-documents" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Document 1 text...", "Document 2 text..."]}'
```

### 6. Использование Python скрипта

```bash
python example_usage.py
```

## Архитектура потока выполнения

1. **User** отправляет запрос через API Gateway
2. **Router Agent** анализирует запрос и определяет подходящего агента
3. Один из специализированных агентов обрабатывает запрос:
   - **Text Reasoning Agent** - для сложных рассуждений
   - **Vision Agent** - для анализа изображений
   - **Retrieval Agent** - для поиска в базе знаний (RAG)
   - **Tool Agent** - для выполнения внешних инструментов
4. **Critic Agent** проверяет качество ответа
5. **Aggregation** объединяет результаты
6. **User** получает финальный ответ

## Настройка компонентов

### Векторная база данных

По умолчанию используется FAISS (локально). Для использования Qdrant:

1. Установите и запустите Qdrant
2. Обновите `.env`:
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Инструменты

Отключить инструменты можно в `.env`:
```env
ENABLE_WEB_SEARCH=false
ENABLE_PYTHON_EXEC=false
```

## Troubleshooting

### Ошибка импорта модулей

Убедитесь, что все зависимости установлены:
```bash
pip install -r requirements.txt
```

### Ошибка API ключа

Проверьте, что файл `.env` создан и содержит правильный `OPENAI_API_KEY`

### Ошибка векторной базы

Если векторная база не инициализирована, система будет работать, но RAG функциональность будет ограничена. Добавьте документы через API.

## Следующие шаги

1. Добавьте свои документы в RAG через API
2. Настройте агентов под ваши задачи
3. Расширьте систему новыми инструментами или агентами
4. Настройте мониторинг и логирование

