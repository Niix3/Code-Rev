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

#### Подсказка

В упрощённой версии проекта доступны только текстовые запросы через `/query`.

### 6. Использование Python скрипта

```bash
python example_usage.py
```

## Архитектура потока выполнения

1. **User** отправляет запрос через API Gateway
2. **Router Agent** анализирует запрос и определяет подходящего агента
3. Один из специализированных агентов обрабатывает запрос:
   - **Text Reasoning Agent** - для сложных рассуждений
   - **Tool Agent** - для выполнения внешних инструментов
4. **Critic Agent** проверяет качество ответа
5. **Aggregation** объединяет результаты
6. **User** получает финальный ответ

## Настройка компонентов


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



## Следующие шаги

1. Добавьте свои документы в RAG через API
2. Настройте агентов под ваши задачи
3. Расширьте систему новыми инструментами или агентами
4. Настройте мониторинг и логирование

