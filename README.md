# AI Assistant

Python-ядро для анализа внешних брифов заказчиков на студенческие проекты
«Мастерской».

Система загружает и нормализует один бриф, извлекает структурированные факты с
помощью LLM, детерминированно проверяет полноту данных, выполняет один
объединённый LLM-этап `Assessment`, применяет настраиваемые правила арбитража,
генерирует уточняющие вопросы по шаблонам, при необходимости формирует MVP-план
и возвращает структурированный JSON.

## Пайплайн

1. `BriefInputFactory` загружает текст или UTF-8 файл и нормализует пробелы.
2. `Extractor` извлекает фактические данные в `ExtractedBrief`.
3. `CompletenessCheckStage` проверяет обязательные поля из `config/criteria.yaml`.
4. `AssessmentStage` оценивает критерии и риски за один LLM-вызов.
5. `DeterministicArbiterStage` вычисляет итоговую рекомендацию по правилам.
6. `TemplateQuestionGeneratorStage` генерирует вопросы без LLM-вызовов.
7. `MVPPlannerStage` вызывает LLM только если Arbiter вернул `SIMPLIFY`.
8. `ResponseWriterStage` детерминированно формирует черновик ответа заказчику.

Типичная стоимость LLM:

- Готовый бриф или бриф с уточнениями: `Extractor` + `Assessment` = 2 LLM-вызова.
- Бриф, которому нужно упрощение: `Extractor` + `Assessment` + `MVPPlanner` = 3 LLM-вызова.

## Конфигурация

Создайте `.env` на основе `.env.example`:

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
DEBUG=false
LOG_LEVEL=INFO
```

Критерии проекта, обязательные поля, типы рисков и правила арбитража находятся здесь:

```text
config/criteria.yaml
```

Шаблоны уточняющих вопросов находятся здесь:

```text
config/question_templates.json
```

Бизнес-критерии намеренно вынесены в конфигурацию и не зашиты жёстко в промпты
или Python-логику принятия решений.

## Запуск

Анализ текста из командной строки:

```bash
python -m app.main --text "Нужно сделать сайт для образовательного проекта..."
```

Анализ UTF-8 файла:

```bash
python -m app.main --file examples/brief.txt
```

Только нормализация ввода без вызова LLM:

```bash
python -m app.main --text "Текст брифа" --normalize-only
```

## Результат

CLI возвращает JSON:

```json
{
  "summary": "Краткое описание проекта",
  "extracted_fields": {
    "goal": "Цель проекта",
    "expected_result": "Ожидаемый результат",
    "tasks": ["Задача 1"],
    "domain": "Предметная область",
    "direction": "development",
    "available_materials": ["Данные", "Макеты"],
    "missing_information": ["Что не указано"],
    "complexity_factors": ["Что усложняет проект"]
  },
  "assessment": {
    "recommendation": "accept",
    "confidence": "high",
    "reasons": ["Почему дана рекомендация"],
    "risks": ["Риски проекта"]
  },
  "clarifying_questions": ["Вопрос заказчику"],
  "mvp_suggestion": "Предложение по упрощению проекта",
  "customer_response_draft": "Черновик ответа заказчику"
}
```

## Тесты

```bash
python -m unittest discover -s tests
```
