# Архитектура pipeline

Этот документ описывает архитектуру приложения по текущему коду репозитория. Источник истины здесь - файлы `app/`, `config/` и `prompts/`.

Приложение - это CLI-инструмент для анализа одного проектного брифа. На вход приходит текст или UTF-8 файл, дальше система:

1. нормализует текст;
2. извлекает факты через LLM;
3. проверяет полноту фактов обычным Python-кодом;
4. делает экспертную LLM-оценку критериев и рисков;
5. детерминированно выбирает итоговый статус по правилам из `criteria.yaml`;
6. генерирует уточняющие вопросы по шаблонам;
7. при необходимости просит LLM предложить MVP;
8. детерминированно собирает итоговый JSON и черновик ответа заказчику.

Главный production pipeline создаётся в `BriefAnalysisPipeline.from_llm_client()` в `app/pipeline/orchestrator.py`. Реальный порядок этапов такой:

```text
СЫРОЙ БРИФ: --text или --file
     │
     ▼
BriefInputFactory [PYTHON]
     │
     ▼
BriefInput
├── original_text
├── normalized_text
└── metadata
     │
     ▼
AIContext.from_brief() [PYTHON]
     │
     ▼
AIContext
├── inputs.brief_input
├── results: пока пустые
└── technical.configuration = CriteriaConfig
     │
     ▼
Extractor [LLM]
     │
     ▼
ExtractionResult
├── extracted_brief: ExtractedBrief
└── technical_info: ExtractorTechnicalInfo
     │
     ▼
AIContext.results.extracted_brief
     │
     ▼
CompletenessCheckStage [PYTHON]
     │
     ▼
CompletenessResult
├── present_information
├── missing_information
├── clarification_information
└── level
     │
     ▼
AssessmentStage [LLM]
     │
     ▼
AssessmentResult
├── criterion_evaluations
├── risks
├── evidence
├── recommendation
└── technical_info
     │
     ▼
DeterministicArbiterStage [PYTHON]
     │
     ▼
ArbitrationResult
├── final_status: ACCEPT / REJECT / CLARIFY / SIMPLIFY / MENTOR_REVIEW
├── reasons
├── evidence
└── triggered_rules
     │
     ▼
TemplateQuestionGeneratorStage [PYTHON]
     │
     ▼
QuestionGenerationResult
├── questions
└── technical_info.llm_invoked = false
     │
     ▼
MVPPlannerStage [CONDITIONAL LLM]
     │
     ├── если final_status != SIMPLIFY
     │   └── MVPPlanningResult(plan=None, llm_invoked=false)
     │
     └── если final_status == SIMPLIFY
         └── LLM -> MVPPlanningResult(plan=MVPPlan, llm_invoked=true)
     │
     ▼
ResponseWriterStage [PYTHON]
     │
     ▼
AIContext.response
├── text: customer_response_draft
└── payload: dict
     │
     ▼
BriefAnalysisResult
```

Важная архитектурная идея: почти все этапы после создания `BriefInput` передают друг другу не отдельные аргументы, а общий `AIContext`. Этот объект похож на контейнер состояния pipeline: каждый этап берёт из него уже готовые данные, добавляет свой результат и возвращает новую версию контекста.

`AIContext` в коде сделан frozen Pydantic-моделью. Это значит, что этапы не мутируют старый объект напрямую, а вызывают методы вида `with_extraction_result()`, `with_assessment_result()`, `with_final_response()`. Такой подход снижает риск случайно испортить состояние предыдущих этапов.

# Точка входа CLI (`app/main.py`)

Этот файл отвечает за запуск приложения из командной строки. Он не содержит бизнес-логики оценки брифа. Его роль - разобрать аргументы CLI, подготовить входной объект, создать LLM-клиент, собрать pipeline и вывести JSON.

```text
Командная строка
     │
     ▼
build_parser()
     │
     ├── --text
     ├── --file
     └── --normalize-only
     │
     ▼
run()
     │
     ├── BriefInputFactory.from_text()
     └── BriefInputFactory.from_file()
     │
     ▼
BriefInput
     │
     ├── если --normalize-only
     │   └── печать BriefInput JSON без LLM
     │
     └── иначе
         ├── Config.load()
         ├── LLMClientFactory.create()
         ├── BriefAnalysisPipeline.from_llm_client()
         ├── pipeline.analyze()
         └── печать BriefAnalysisResult JSON
```

## Импорты

### Стандартная библиотека

`argparse` используется для описания CLI-аргументов. В этом файле он задаёт взаимоисключающий выбор: пользователь обязан передать либо `--text`, либо `--file`.

`json` нужен только на границе приложения: результат Pydantic-моделей превращается в человекочитаемый JSON через `model_dump(mode="json")`.

`sys` нужен для доступа к `sys.argv` и печати ошибок в `stderr`.

`Path` используется для аргумента `--file`, чтобы путь к файлу сразу был объектом, а не простой строкой.

`Sequence` - type hint. Он показывает, что `run()` может принять последовательность строк, например список аргументов в тестах.

### Импорты проекта

`Config` загружает `.env` и настройки LLM.

`BriefInputFactory` создаёт нормализованный `BriefInput`.

`BriefInputError` ловится отдельно, потому что ошибки входного текста надо показать как ошибку CLI-аргументов.

`LLMClientFactory` создаёт конкретный LLM-клиент. Сейчас это OpenRouter-клиент.

`BriefAnalysisPipeline` - главный pipeline анализа.

## Ключевые функции

### `build_parser()`

Метод строит объект `ArgumentParser`. Здесь важен вызов `add_mutually_exclusive_group(required=True)`: он запрещает одновременно передать `--text` и `--file`, но требует выбрать один источник.

Это детерминированная Python-логика. LLM здесь не вызывается.

### `run(argv: Sequence[str] | None = None) -> int`

Это главный сценарий CLI.

Метод получает список аргументов или берёт аргументы из реальной командной строки. Затем создаёт `BriefInputFactory` и строит `BriefInput`.

Если `BriefInputFactory` выбрасывает `BriefInputError`, `parser.error()` печатает понятную ошибку и приложение возвращает код `2`. Это стандартная практика CLI: код `2` обычно означает неправильный ввод пользователя.

Если указан `--normalize-only`, pipeline вообще не запускается. Это полезный диагностический режим: можно проверить, как приложение очистит текст, не тратя LLM-вызовы.

Если нужен полный анализ, метод:

1. загружает настройки через `Config.load()`;
2. создаёт LLM-клиент через `LLMClientFactory.create(settings)`;
3. собирает pipeline через `BriefAnalysisPipeline.from_llm_client(...)`;
4. запускает `pipeline.analyze(brief_input)`;
5. печатает результат как JSON.

`try/except` вокруг pipeline ловит `RuntimeError` и `BriefAnalysisPipelineError`. Это граница приложения: внутренние ошибки превращаются в сообщение `Pipeline error: ...` и код возврата `1`.

### Где здесь используется LLM

В `app/main.py` LLM напрямую не вызывается. Файл только создаёт LLM-клиент и передаёт его в pipeline. Реальные вызовы происходят глубже: в `Extractor`, `AssessmentStage` и условно в `MVPPlannerStage`.

# Вход брифа (`app/input/brief_input.py`)

Этот файл отвечает за подготовку сырого текста к pipeline. Он принимает текст напрямую или читает файл, убирает технический шум и создаёт `BriefInput`.

```text
                        СЫРОЙ БРИФ
                             │
                ┌────────────┴────────────┐
                │                         │
              текст                      файл
                │                         │
                │                 BriefInputFactory
                │                      .from_file()
                │                         │
                └────────────┬────────────┘
                             │
                    BriefInputFactory
                         .from_text()
                             │
                    BriefInputNormalizer
                         .normalize()
                             │
                      normalized_text
                             │
                         BriefInput
```

## Импорты

### Стандартная библиотека

`re` - модуль регулярных выражений. В этом файле он нужен для поиска невидимых символов, нескольких пробелов подряд и начальных пробелов строки.

`Path` используется для чтения файла в `from_file()`.

`from __future__ import annotations` откладывает вычисление type hints. Для junior-разработчика важно понять: это не меняет runtime-логику нормализации, а помогает Python проще работать с аннотациями типов.

### Импорты проекта

`BriefInput` и `BriefInputMetadata` приходят из `app.schemas`. Это Pydantic-модели входного слоя.

## Константы и служебные конструкции

`_ZERO_WIDTH_CHARS` - regex для невидимых символов: zero-width space, BOM и похожие символы. В брифе они не несут бизнес-смысла, но могут ломать сравнения и проверки пустоты.

`_MULTIPLE_SPACES` - regex для двух и более пробелов или табов подряд. Он используется внутри строки, чтобы заменить технически разные варианты пробелов одним пробелом.

`_WRAPPER_MARKERS` - набор строк вроде `<<< BEGIN OF TEXT >>>`. Такие маркеры часто появляются при копировании текста из тестовых обёрток или чатов. Нормализатор удаляет строки, которые полностью совпадают с этими маркерами.

Здесь используется `set`, а не список, потому что проверка `line.strip() in _WRAPPER_MARKERS` для множества обычно быстрее и семантически точнее: нам важен факт принадлежности, а не порядок.

## Классы

### `BriefInputError`

Это собственный тип ошибки для проблем подготовки брифа.

```python
class BriefInputError(RuntimeError):
    pass
```

В коде тело класса фактически пустое, но класс уже работает как исключение благодаря наследованию от `RuntimeError`.

Зачем нужен отдельный тип:

- пустой текст;
- текст стал пустым после очистки;
- файл невозможно прочитать;
- файл не UTF-8.

Причины разные, но для входного слоя это одна архитектурная категория: "не удалось подготовить бриф".

### `BriefInputNormalizer`

Это детерминированный нормализатор текста.

Он не должен менять смысл брифа. Его задача - убрать технические различия, которые не важны для анализа: Windows/Linux переносы строк, нулевые байты, невидимые символы, лишние пробелы, повторяющиеся пустые строки.

Основной метод - `normalize(text: str) -> str`.

Что получает: строку с сырым текстом.

Что делает:

1. проверяет, что текст не `None` и не пустой после `strip()`;
2. приводит `\r\n` и `\r` к `\n`;
3. удаляет `\x00`;
4. удаляет zero-width символы;
5. проходит по строкам;
6. удаляет wrapper-маркеры;
7. схлопывает повторные пустые строки;
8. сохраняет ведущие пробелы строки, но нормализует лишние пробелы внутри содержательной части;
9. удаляет пустые строки в начале и конце;
10. проверяет, что результат не пуст.

Почему сохраняются ведущие пробелы: иногда в тексте может быть список или вложенный блок. Код не пытается полностью переформатировать текст, он только убирает шум.

Важные Python-конструкции:

- `for raw_line in normalized.split("\n")` - цикл по строкам;
- `continue` - ранний переход к следующей строке, когда текущую уже обработали;
- `while lines and not lines[0].strip()` - удаление пустых строк с начала без ошибки на пустом списке;
- `re.match(...).group(0)` - получение ведущих пробелов.

Что возвращает: нормализованную строку.

Куда результат идёт дальше: в `BriefInput.normalized_text`, затем в `Extractor` и `AssessmentStage`.

### `BriefInputFactory`

Это factory-класс. Factory означает, что объект не просто хранит данные, а создаёт другие объекты по правилам.

Зачем выделять factory отдельно:

- создание `BriefInput` из текста и из файла имеет общую логику;
- нормализатор можно подменить в тестах или при расширении;
- CLI не должен знать детали очистки текста.

`__init__(normalizer: BriefInputNormalizer | None = None)` принимает зависимость. Если нормализатор не передали, создаётся стандартный `BriefInputNormalizer`. Это простая форма dependency injection: класс получает зависимость снаружи, но умеет создать дефолтную.

`from_text(text, metadata=None)`:

- проверяет текст через `_ensure_text()`;
- нормализует его;
- создаёт `BriefInput`;
- если metadata не передали, создаёт `BriefInputMetadata()`.

`from_file(file_path, metadata=None)`:

- превращает путь в `Path`;
- читает файл как UTF-8;
- при `OSError` или `UnicodeDecodeError` выбрасывает `BriefInputError`;
- дополняет metadata значениями `source="file"`, `input_type="file"`, `file_path`, `file_name`;
- дальше вызывает `from_text()`.

`@staticmethod` у `_ensure_text()` означает: метод логически относится к классу, но не использует `self` и не зависит от состояния объекта.

### Где здесь используется LLM

Этот этап детерминированный: LLM здесь не вызывается.

# Модели входа (`app/schemas/brief.py`)

Этот файл описывает Pydantic-модели, которые проходят между входным слоем и pipeline.

```text
BriefInput
├── original_text: str
├── normalized_text: str
└── metadata: BriefInputMetadata
    ├── source: str
    ├── input_type: str
    ├── file_path: str | None
    ├── file_name: str | None
    ├── encoding: str
    └── extra: dict[str, Any]
```

`BaseModel` из Pydantic даёт валидацию типов, сериализацию в JSON и удобное копирование через `model_copy()`.

`Field(default_factory=dict)` важен для mutable-значений. Если написать `extra: dict = {}`, один и тот же словарь мог бы переиспользоваться между объектами. `default_factory` создаёт новый словарь для каждого экземпляра.

`ConfigDict(extra="forbid")` в `BriefInput` запрещает лишние поля. Это полезно для контрактов pipeline: если кто-то случайно передаст поле с опечаткой, Pydantic не проглотит его молча.

`ConfigDict(extra="allow")` в `BriefInputMetadata` наоборот разрешает дополнительные поля. Metadata - технический слой, его можно расширять без изменения основной модели.

`@field_validator("original_text", "normalized_text")` проверяет, что обе строки не пустые.

`str | None` - современная запись "строка или None". То же самое, что старый стиль `Optional[str]`.

# Общий контейнер pipeline (`app/schemas/ai_context.py`)

`AIContext` - центральная структура данных production pipeline. Каждый stage получает `AIContext` и возвращает новый `AIContext`.

```text
AIContext
├── inputs: PipelineInputState
│   └── brief_input: BriefInput
├── retrieval: RetrievalState
│   └── results: list[SearchResult]
├── results: PipelineResults
│   ├── extracted_brief: ExtractedBrief | None
│   ├── extraction_result: ExtractionResult | None
│   ├── completeness_result: CompletenessResult | None
│   ├── assessment_result: AssessmentResult | None
│   ├── arbitration_result: ArbitrationResult | None
│   ├── clarification_result: QuestionGenerationResult | None
│   ├── mvp_planning_result: MVPPlanningResult | None
│   └── self_check_result: SelfCheckResult | None
├── response: ResponseState
│   ├── text: str | None
│   └── payload: dict[str, Any] | None
└── technical: PipelineTechnicalState
    ├── metadata: dict[str, Any]
    ├── configuration: CriteriaConfig | None
    └── stage_metadata: dict[str, dict[str, Any]]
```

`AIContext.from_brief()` создаёт стартовое состояние: вход уже есть, результаты ещё пустые, конфигурация может быть сохранена в `technical.configuration`.

Методы `with_*` возвращают новую версию контекста:

- `with_extraction_result()` кладёт и полный `ExtractionResult`, и короткий доступ `extracted_brief`;
- `with_completeness_result()` добавляет результат проверки полноты;
- `with_assessment_result()` добавляет LLM-оценку;
- `with_arbitration_result()` добавляет финальный статус арбитража;
- `with_clarification_result()` добавляет вопросы;
- `with_mvp_planning_result()` добавляет результат MVP-планирования;
- `with_final_response()` добавляет текст и финальный payload.

Почему так сделано: pipeline становится линейным и предсказуемым. Каждый этап знает: "я беру контекст, проверяю нужные поля, добавляю свой результат".

Важная Python/Pydantic-конструкция: `model_config = ConfigDict(extra="forbid", frozen=True)`. `frozen=True` делает модель иммутабельной на уровне Pydantic: её нельзя случайно изменить присваиванием. Поэтому код использует `model_copy(update=...)`.

`self` в методах - ссылка на текущий объект. Например, `self.results` означает "results у этого конкретного AIContext".

# Сборка pipeline (`app/pipeline/orchestrator.py`)

Этот файл отвечает за композицию этапов. Он не анализирует бриф сам, а соединяет компоненты в нужном порядке.

```text
BriefAnalysisPipeline.from_llm_client(llm_client)
     │
     ├── tracing = get_tracing_client()
     ├── config = get_criteria_config()
     ├── prompts = PromptManager()
     ├── llm_runner = LLMRunner(...)
     │
     └── stages:
         1. Extractor
         2. CompletenessCheckStage
         3. AssessmentStage
         4. DeterministicArbiterStage
         5. TemplateQuestionGeneratorStage
         6. MVPPlannerStage
         7. ResponseWriterStage
```

## Импорты

`Sequence` используется для списка stage-объектов.

`Protocol` используется для `ContextStage`. Protocol описывает не наследование, а форму объекта: "любой объект с методом `run_context(context) -> AIContext` подходит". Это удобно для pipeline, потому что этапы могут быть разными классами, но иметь один контракт.

`CriteriaConfig`, `get_criteria_config` дают общий конфиг критериев.

`BriefInputFactory` нужен для метода `analyze_text()`, который принимает строку и сам превращает её в `BriefInput`.

`LLMRunner` создаётся один раз и переиспользуется LLM-этапами. Это значит, что retry, timeout, модель и tracing единообразны.

## Классы

### `BriefAnalysisPipeline`

Это главный orchestration-класс.

Он получает:

- список stages;
- конфигурацию критериев;
- input factory.

Если stages не передали, конструктор создаёт пустой список. В production-коде обычно используется `from_llm_client()`, который наполняет список реальными этапами.

### `from_llm_client()`

Это classmethod. `@classmethod` получает первым параметром не объект `self`, а класс `cls`. Поэтому метод может создать новый `BriefAnalysisPipeline`, не имея уже готового экземпляра.

Метод принимает `llm_client`, а не создаёт его сам. Это dependency injection: внешний слой решает, какой клиент использовать, а pipeline работает с абстрактным интерфейсом.

Важно: `Extractor`, `AssessmentStage` и `MVPPlannerStage` получают один и тот же `LLMRunner`. Остальные этапы LLM не используют.

### `analyze(brief_input)`

Получает `BriefInput`, запускает `run_context()`, затем проверяет, что `context.final_response_payload` не `None`.

После этого вызывает `BriefAnalysisResult.model_validate(context.final_response_payload)`. То есть финальный словарь ещё раз превращается в строго типизированную Pydantic-модель.

### `run_context(brief_input)`

Создаёт стартовый `AIContext.from_brief(...)`, затем обычным циклом:

```python
for stage in self._stages:
    context = stage.run_context(context)
```

Этот цикл - сердце pipeline. Каждый этап получает результат предыдущего.

### Где здесь используется LLM

`orchestrator.py` сам LLM не вызывает, но создаёт `LLMRunner` и передаёт его LLM-этапам.

# Базовые stage-классы (`app/pipeline/contracts.py`, `app/pipeline/base.py`)

Эти файлы задают общий каркас выполнения этапов.

```text
BaseStage.run(stage_input)
     │
     ├── _build_trace_input()
     ├── tracing_client.create_trace()
     ├── _run(stage_input)       <-- реализует конкретный stage
     ├── _build_trace_output()
     └── return result

BaseLLMStage._run(stage_input)
     │
     ├── build_prompt()
     ├── build_system_prompt()
     ├── build_context()
     ├── LLMRunner.run()
     ├── validate_payload()
     └── postprocess()
```

## `BaseStage`

`BaseStage` - абстрактный базовый класс для этапов. Он наследуется от `ABC`, а метод `_run()` помечен `@abstractmethod`. Это означает: нельзя полноценно использовать `BaseStage` сам по себе, конкретный наследник обязан реализовать `_run()`.

`BaseStage.run()` даёт всем этапам одинаковое поведение:

- логирование старта;
- создание trace;
- запуск конкретной логики `_run()`;
- обработка исключений;
- логирование успеха;
- запись trace output.

Так конкретным этапам не нужно каждый раз вручную писать одинаковый код вокруг своей бизнес-логики.

`StageExecutionError` - общий тип ошибки этапа. Если конкретный stage не переопределил `_build_stage_exception()`, обычная ошибка будет обёрнута в `StageExecutionError`.

## `BaseLLMStage`

`BaseLLMStage` расширяет `BaseStage` для этапов, которые вызывают языковую модель.

Он отвечает за:

- загрузку prompt-файла через `PromptManager`;
- выбор output Pydantic-модели;
- вызов `LLMRunner`;
- postprocess результата;
- единообразные ошибки LLM-этапов.

`Generic[TInput, TPayload, TOutput]` - type hints для обобщённого класса. Они помогают указать: stage получает один тип, LLM возвращает payload другого типа, а наружу stage может вернуть третий тип.

Например, `Extractor` объявлен как:

```python
BaseLLMStage[BriefInput, ExtractedBrief, ExtractionResult]
```

Это значит:

- вход stage: `BriefInput`;
- структурированный ответ LLM: `ExtractedBrief`;
- итог метода `run()`: `ExtractionResult`.

`output_model` - class variable. Конкретный LLM-stage задаёт Pydantic-модель, по JSON Schema которой нужно валидировать ответ LLM.

### Где здесь используется LLM

`BaseLLMStage._run()` вызывает `self._llm_runner.run(...)`. Это общий механизм LLM-вызовов для `Extractor` и `AssessmentStage`.

`_execute_structured_stage()` вызывает `LLMRunner.run_json(...)`. В текущем production pipeline он используется в `MVPPlannerStage`.

# LLM-инфраструктура (`app/llm/client.py`, `app/llm/factory.py`, `app/llm/openrouter.py`, `app/llm/runner.py`)

Эти файлы отделяют бизнес-логику от конкретного LLM-провайдера.

```text
Extractor / AssessmentStage / MVPPlannerStage
     │
     ▼
LLMRunner
     │
     ├── строит messages
     ├── добавляет JSON Schema инструкцию
     ├── вызывает LLMClient.generate_json()
     ├── валидирует Pydantic-модель
     ├── делает retry
     └── возвращает LLMRunResult
     │
     ▼
OpenRouterLLMClient
     │
     ▼
OpenAI SDK с base_url=https://openrouter.ai/api/v1
```

## `LLMClient`

Это абстрактный интерфейс. Он говорит: любой LLM-клиент в проекте должен уметь:

- `generate()` - вернуть текст;
- `generate_json()` - вернуть JSON-объект как `dict`;
- `stream()` - отдавать текст частями.

В production pipeline реально используется `generate_json()`.

## `LLMClientFactory`

Сейчас factory всегда возвращает `OpenRouterLLMClient(settings=settings)`.

Зачем factory: если позже появится другой провайдер, точка замены будет здесь, а pipeline не придётся переписывать.

## `OpenRouterLLMClient`

Этот класс реализует `LLMClient` через OpenAI SDK, но с `base_url` OpenRouter.

Он получает `Settings`, берёт из них `openrouter_api_key` и `openrouter_model`, создаёт `OpenAI(...)`.

Секреты в документации не раскрываются. В коде они берутся из `.env` через `Settings`.

`generate_json()` вызывает `generate()`, затем делает `json.loads(response_text)`. Если модель вернула невалидный JSON или JSON не является объектом, метод выбрасывает `RuntimeError`.

## `LLMRunner`

Это главный технический слой LLM-вызовов.

Что получает:

- `llm_client`;
- tracing client;
- logger;
- `max_retries`;
- `timeout_seconds`;
- `model_name`.

Что делает `run_json()`:

1. добавляет инструкцию вернуть JSON-объект;
2. добавляет JSON Schema ожидаемой Pydantic-модели;
3. создаёт trace;
4. запускает попытки от `1` до `max_retries`;
5. вызывает `_call_generate_json()`;
6. валидирует ответ через `response_model.model_validate(raw_response)`;
7. дополнительно вызывает `payload_validator`, если он передан;
8. собирает latency, token usage и metadata;
9. возвращает `LLMRunResult`.

Если все попытки закончились ошибкой, выбрасывается `LLMRunnerProviderError`.

`ThreadPoolExecutor` используется для timeout. LLM-клиент сам по себе может зависнуть на сетевом вызове; runner запускает вызов в отдельном worker-потоке и ждёт `future.result(timeout=...)`.

`_ensure_json_instruction()` добавляет к первому message текст:

- вернуть только валидный JSON object;
- все человекочитаемые строки в JSON должны быть на русском;
- JSON должен соответствовать JSON Schema.

Это важно: LLM не просто "попросили ответить JSON", ей передают схему Pydantic-модели.

### Где здесь используется LLM

Реальный сетевой вызов происходит в `OpenRouterLLMClient.generate()`, когда вызывается `self._client.chat.completions.create(...)`.

# Prompt management (`app/prompts/manager.py`, `prompts/*.md`)

`PromptManager` отвечает за загрузку и рендер prompt-файлов.

```text
BaseLLMStage
     │
     ▼
PromptManager.load(prompt_name)
     │
     ├── ищет .md файл
     ├── читает front matter
     └── кэширует Prompt
     │
     ▼
PromptManager.render(...)
     │
     ├── делит файл на System/User секции
     ├── заменяет {{variables}}
     └── возвращает RenderedPrompt
```

`_TEMPLATE_VARIABLE_RE` - regex для переменных вида `{{ brief_text }}`. Он находит имя переменной и подставляет значение из словаря.

Prompt-файлы имеют YAML-like front matter между `---`. Например, `prompts/extractor.md` содержит `name`, `version`, `variables`, `output_model`.

`PromptManager` не использует полноценную YAML-библиотеку для prompt metadata. Он парсит простые `key: value` строки сам.

`_split_prompt_sections()` ищет секции `# System` и `# User`. Если секции `System` нет, весь prompt считается system-текстом.

`_stringify_variable()` превращает значения в строку. Если значение - Pydantic-модель, используется `model_dump_json()`. Если dict/list/tuple - `json.dumps(..., ensure_ascii=False, indent=2)`.

# Извлечение фактов (`app/pipeline/extractor.py`)

Этот файл отвечает за первый LLM-этап: перевод неструктурированного текста брифа в строго типизированный `ExtractedBrief`.

```text
AIContext
└── brief_input: BriefInput
    └── normalized_text
          │
          ▼
Extractor.extract_context()
          │
          ▼
Extractor.extract()
          │
          ▼
BaseLLMStage.run()
          │
          ▼
prompts/extractor.md + {{brief_text}}
          │
          ▼
LLMRunner.run(output_model=ExtractedBrief)
          │
          ▼
ExtractedBrief
          │
          ▼
Extractor._normalize_extracted_brief()
          │
          ▼
ExtractionResult
├── extracted_brief
└── technical_info
          │
          ▼
AIContext.with_extraction_result()
```

## Импорты

`Path` нужен для пути к prompt-файлу по умолчанию.

`ClassVar` используется для `output_model`. Это поле класса, а не поле экземпляра.

`TYPE_CHECKING` позволяет импортировать `LLMClient` только для type hints. Во время выполнения этот импорт не нужен.

`BaseLLMStage` даёт общую LLM-обвязку.

`BriefInput`, `ExtractedBrief`, `ExtractionResult`, `ExtractedFact`, `ExtractorTechnicalInfo`, `AIContext` - модели данных, которые stage получает и возвращает.

## Класс `Extractor`

### Что это

`Extractor` - LLM-stage, который извлекает факты из брифа. Он не должен оценивать качество проекта, искать риски или принимать решение.

### Зачем он нужен

Неструктурированный текст неудобен для детерминированных проверок. Чтобы Python-код дальше мог проверить "есть ли цель", "есть ли задачи", "есть ли материалы", LLM сначала приводит текст к модели `ExtractedBrief`.

### Что получает

Основной вход - `BriefInput`, особенно `brief_input.normalized_text`.

В pipeline он получает `AIContext`, достаёт из него `brief_input`, запускает extract и возвращает контекст с результатом.

### Что делает

1. Рендерит `prompts/extractor.md`, подставляя `brief_text`.
2. Передаёт prompt в `LLMRunner`.
3. Ожидает JSON, соответствующий `ExtractedBrief`.
4. Нормализует каждое поле `ExtractedFact`: чистит `value`, `evidence`, `notes`.
5. Оборачивает результат в `ExtractionResult`.

### Что возвращает

`ExtractionResult`:

- `extracted_brief`;
- `technical_info` с числом попыток, именем prompt, trace, model, raw response, recovered errors.

### Куда результат идёт дальше

`CompletenessCheckStage` берёт `context.extracted_brief` и сверяет его с обязательными полями из `criteria.yaml`.

## Модели данных extraction (`app/schemas/extraction.py`)

```text
ExtractedFact
├── status: explicit / missing / uncertain
├── value: str | None
├── evidence: list[str]
├── confidence: float | None
└── notes: str | None

ExtractedBrief
├── project_goal: ExtractedFact
├── tasks: list[ExtractedFact]
├── project_type: ExtractedFact
├── project_direction: ExtractedFact
├── technologies: list[ExtractedFact]
├── stack: list[ExtractedFact]
├── materials: list[ExtractedFact]
├── expected_result: ExtractedFact
├── constraints: list[ExtractedFact]
├── deadlines: list[ExtractedFact]
├── existing_resources: list[ExtractedFact]
├── integrations: list[ExtractedFact]
└── other_facts: list[ExtractedFact]
```

`FactStatus` - enum. Enum ограничивает допустимые строки. Вместо произвольного `"yes"` или `"found"` код принимает только `explicit`, `missing`, `uncertain`.

`ExtractedFact.value` имеет validator: пустая строка превращается в `None`. Это помогает дальнейшим проверкам не отличать "нет значения" от "строка из пробелов".

### Где здесь используется LLM

LLM вызывается в `Extractor` через `BaseLLMStage._run()` и `LLMRunner.run()`.

Prompt: `prompts/extractor.md`.

Данные, передаваемые модели: нормализованный текст брифа в переменной `brief_text`.

Ожидаемый результат: JSON по схеме `ExtractedBrief`.

Что происходит дальше: результат нормализуется, кладётся в `ExtractionResult`, затем в `AIContext.results`.

# Проверка полноты (`app/pipeline/completeness.py`)

Этот файл отвечает за детерминированную проверку: достаточно ли в `ExtractedBrief` обязательной информации.

```text
AIContext
└── extracted_brief: ExtractedBrief
          │
          ▼
CompletenessCheckStage.check_context()
          │
          ▼
criteria.yaml
└── evaluation.required_fields
          │
          ▼
for field_def in required_fields
          │
          ├── _resolve_field_path()
          ├── _classify_value()
          ├── present / missing / clarification
          └── CompletenessItem
          │
          ▼
CompletenessResult
          │
          ▼
AIContext.with_completeness_result()
```

## Импорты

`logging` нужен для передачи logger в `BaseStage`.

`Any`, `get_args`, `get_origin` используются для анализа type hints при валидации `field_path`.

`BaseModel` из Pydantic нужен, потому что `_resolve_field_path()` умеет идти по Pydantic-моделям.

`CriteriaConfig`, `CriteriaLoader`, `RequiredField`, `get_criteria_config` связывают stage с `config/criteria.yaml`.

`CompletenessResult`, `CompletenessItem`, `CompletenessStatus`, `CompletenessLevel`, `ExtractedFact`, `FactStatus` - схемы результата и входных фактов.

## Класс `CompletenessCheckStage`

### Что это

Детерминированный stage без LLM. Он проверяет не качество проекта, а наличие нужных данных.

### Зачем он нужен

LLM может извлечь факты, но решение "обязательное поле есть или нет" лучше делать обычным кодом. Это делает поведение повторяемым и объяснимым.

### Что получает

`AIContext` с заполненным `extracted_brief`.

Если `extracted_brief` отсутствует, stage выбрасывает `CompletenessError`.

### Что делает

1. Загружает `CriteriaConfig`.
2. Проверяет, что в конфиге есть `evaluation.required_fields`.
3. Проверяет, что каждый `field_path` реально существует в `ExtractedBrief`.
4. Для каждого required field достаёт значение из `ExtractedBrief`.
5. Классифицирует значение как `present`, `missing` или `clarification`.
6. Собирает списки present/missing/critical/clarification.
7. Вычисляет общий `level`.

### Как работает классификация

Если значение - `ExtractedFact`:

- `status == explicit` и `value is not None` -> `present`;
- если это поле `project_type`, дополнительно проверяется, что тип известен по `criteria.yaml`;
- `status == uncertain` -> `clarification`;
- иначе -> `missing`.

Если значение - список `ExtractedFact`:

- пустой список -> `missing`;
- все значения explicit, нет uncertain и missing -> `present`;
- есть explicit или uncertain, но список неоднозначный -> `clarification`;
- иначе -> `missing`.

Если optional field отсутствует, stage не добавляет его в результат. Поэтому `missing_information` содержит только обязательные пропуски.

### Что возвращает

`CompletenessResult`:

```text
CompletenessResult
├── is_complete: bool
├── level: complete / needs_clarification / incomplete
├── missing_information: list[CompletenessItem]
├── critical_missing_information: list[CompletenessItem]
├── present_information: list[CompletenessItem]
├── clarification_information: list[CompletenessItem]
├── warnings: list[str]
└── technical_info
```

`is_complete` становится `True` только если нет ни missing, ни clarification.

### Связь с `criteria.yaml`

В текущем конфиге обязательные поля:

- `project_goal`;
- `expected_result`;
- `tasks`;
- `project_direction`.

Необязательные поля:

- `materials`;
- `deadlines`;
- `integrations`.

Каждый элемент содержит `field_path`. Например, `field_path: project_goal` означает взять `extracted_brief.project_goal`.

### Где здесь используется LLM

Этот этап детерминированный: LLM здесь не вызывается.

# Оценка и анализ рисков (`app/pipeline/assessment.py`)

Этот файл отвечает за второй основной LLM-вызов. Он оценивает проект по критериям и рискам, но не принимает финальное решение.

```text
AIContext
├── normalized_text
├── extracted_brief
└── completeness_result
     │
     ▼
AssessmentPreparation.prepare()
     │
     ├── validate context
     ├── load CriteriaConfig
     ├── criteria = config.evaluation.criteria
     ├── risk_types = config.evaluation.risk_analysis.risk_types
     └── optional retriever.retrieve()
     │
     ▼
AssessmentPreparedInput
     │
     ▼
AssessmentStage.run()
     │
     ▼
prompts/assessment.md
     │
     ▼
LLMRunner.run(output_model=AssessmentPayload)
     │
     ▼
AssessmentPayload
     │
     ▼
AssessmentResult
     │
     ▼
AIContext.with_assessment_result()
```

## Импорты

`Protocol` используется для `AssessmentRetriever`: stage может работать с любым объектом, у которого есть метод `retrieve(...)`.

`Mapping` нужен для metadata filters retriever-а.

`CriteriaConfig`, `Criterion`, `RiskType` связывают LLM-оценку с конфигом.

`SearchResult` - модель опционального retrieval-контекста.

`CriterionEvaluation` и `Risk` - структурированные элементы, которые LLM должна вернуть.

## `AssessmentRetriever`

Это Protocol для optional retrieval. В production-сборке `BriefAnalysisPipeline.from_llm_client()` параметр `retriever` по умолчанию `None`, поэтому retrieval обычно не используется.

Если retriever передан, `AssessmentPreparation` строит поисковый запрос из:

- normalized brief;
- project goal;
- tasks;
- technologies;
- integrations;
- ключей missing information.

Потом результат retrieval добавляется в prompt как `retrieved_context`.

## `AssessmentPreparation`

Это подготовительный класс. Он не вызывает LLM.

Зачем нужен отдельно: подготовка assessment сложнее, чем просто "взять контекст". Нужно проверить, что previous stages уже отработали, загрузить критерии, риск-типы и optional retrieval.

`_validate_context()` требует:

- `context.extracted_brief is not None`;
- `context.completeness_result is not None`.

Если этих данных нет, assessment не имеет права стартовать.

## `AssessmentStage`

### Что это

LLM-stage для аналитической оценки проекта.

### Зачем он нужен

Некоторые вещи трудно надёжно выразить простыми правилами: широкий scope, риск production criticality, пригодность для студентов, достаточность материалов. Поэтому этот stage отдаёт структурированный контекст LLM и просит оценить критерии и риски.

При этом LLM не принимает финальное решение. В prompt прямо сказано: "Do not make a final ACCEPT or REJECT decision." Финальный статус выбирает `DeterministicArbiterStage`.

### Что получает

`AssessmentPreparedInput`, внутри которого:

- `context`;
- `criteria_config`;
- список `criteria`;
- список `risk_types`;
- `retrieved_context`;
- параметры retrieval.

### Что делает

1. Рендерит `prompts/assessment.md`.
2. Передаёт в prompt:
   - normalized brief;
   - extracted facts;
   - completeness result;
   - criteria;
   - risk types;
   - retrieved context.
3. Запускает LLM через `LLMRunner`.
4. Ожидает `AssessmentPayload`.
5. Нормализует текстовые поля в критериях, рисках и evidence.
6. Возвращает `AssessmentResult` с technical info.

### Модели assessment (`app/schemas/assessment.py`, `app/schemas/evaluation.py`, `app/schemas/risk.py`)

```text
AssessmentPayload
├── criterion_evaluations: list[CriterionEvaluation]
├── risks: list[Risk]
├── evidence: list[AssessmentEvidence]
├── has_risks: bool
├── recommendation: ready_for_arbitration / needs_clarification / high_risk_review
├── summary: str | None
└── confidence: float | None
```

`AssessmentResult` содержит те же бизнес-поля плюс `technical_info`.

```text
CriterionEvaluation
├── criterion: str
├── criterion_title: str | None
├── status: met / not_met / insufficient_information / risk_detected
├── evidence: list[str]
├── explanation: str | None
├── confidence: float | None
└── notes: str | None
```

```text
Risk
├── type: str
├── description: str
├── severity: low / medium / high / critical
├── evidence: list[str]
├── confidence: float | None
└── notes: str | None
```

`@model_validator(mode="after")` в `AssessmentPayload` и `AssessmentResult` проверяет согласованность: `has_risks` должен совпадать с тем, есть ли элементы в `risks`.

### Где здесь используется LLM

LLM вызывается в `AssessmentStage`.

Prompt: `prompts/assessment.md`.

Данные, передаваемые модели:

- `normalized_brief`;
- `extracted_brief`;
- `completeness_result`;
- `criteria`;
- `risk_types`;
- `retrieved_context`.

Ожидаемый результат: JSON по схеме `AssessmentPayload`.

Что происходит дальше: `AssessmentResult` передаётся в `DeterministicArbiterStage`.

# Арбитраж (`app/pipeline/arbiter.py`)

Это самый важный файл бизнес-логики принятия решения. Он не вызывает LLM. Он берёт результаты предыдущих этапов и применяет правила из `criteria.yaml`.

```text
AIContext
├── completeness_result
└── assessment_result
    ├── risks
    └── criterion_evaluations
          │
          ▼
DeterministicArbiterStage.arbitrate_context()
          │
          ▼
_build_signals()
          │
          ▼
signals
├── completeness.is_complete
├── completeness.missing_count
├── risk.max_severity
├── risk.high_count
├── evaluation.not_met_count
└── ...
          │
          ▼
for rule in criteria.yaml arbitration.rules
          │
          ├── _rule_matches()
          ├── _condition_matches()
          └── первое совпавшее правило
          │
          ▼
ArbitrationResult
          │
          ▼
AIContext.with_arbitration_result()
```

## Импорты

`ArbitrationConfiguration`, `ArbitrationCondition`, `ArbitrationRule` - Pydantic-модели правил из `criteria.yaml`.

`DecisionStatus` - enum финальных статусов.

`RiskSeverity`, `CriterionEvaluationStatus` нужны для подсчёта сигналов.

## Класс `DeterministicArbiterStage`

### Что это

Детерминированный "судья" pipeline. Он превращает аналитические результаты в финальный статус:

- `ACCEPT`;
- `REJECT`;
- `CLARIFY`;
- `SIMPLIFY`;
- `MENTOR_REVIEW`.

### Зачем он нужен

LLM хорошо делает анализ, но финальное решение должно быть повторяемым. Если одна и та же комбинация рисков и missing fields пришла на вход, arbiter должен вернуть один и тот же статус.

### Что получает

`AIContext` с:

- `completeness_result`;
- `assessment_result`.

Если чего-то нет, выбрасывается `ArbitrationError`.

### Что делает

1. Строит signals.
2. Строит evidence map.
3. Перебирает правила `self._arbitration.rules` в порядке из YAML.
4. Для каждого правила проверяет все conditions.
5. Первое совпавшее правило превращает в `ArbitrationResult`.
6. Если ничего не совпало, берёт `default_status` из конфига.

Порядок правил в `criteria.yaml` задаёт приоритет. Это важно: если одновременно есть critical risk и missing information, победит правило, которое стоит выше.

### Сигналы

Arbiter поддерживает только фиксированный набор сигналов. Конфиг не может сослаться на произвольное поле.

Основные группы:

```text
completeness.*
├── is_complete
├── missing_count
├── clarification_count
└── present_count

risk.*
├── has_risks
├── total_count
├── low_count
├── medium_count
├── high_count
├── critical_count
└── max_severity

evaluation.*
├── total_count
├── met_count
├── not_met_count
├── insufficient_information_count
└── risk_detected_count
```

### Реальные правила из `criteria.yaml`

Текущий порядок:

1. `reject_critical_risk`
   - если `risk.max_severity in ["critical"]`;
   - статус `REJECT`;
   - приоритет самый высокий.

2. `simplify_high_risk`
   - если `risk.max_severity in ["high"]`;
   - статус `SIMPLIFY`.

3. `clarify_missing_information`
   - если `completeness.missing_count > 0`;
   - статус `CLARIFY`.

4. `mentor_review_insufficient_information`
   - если `evaluation.insufficient_information_count > 0`;
   - и `completeness.missing_count == 0`;
   - статус `MENTOR_REVIEW`.

5. `accept_ready`
   - если бриф complete;
   - max risk `none` или `low`;
   - нет `not_met`;
   - нет `insufficient_information`;
   - нет `risk_detected`;
   - статус `ACCEPT`.

Если ничего не совпало, default status - `MENTOR_REVIEW`.

### Приоритеты решений

Фактический приоритет такой:

```text
critical risk
    ↓
REJECT

иначе high risk
    ↓
SIMPLIFY

иначе missing required information
    ↓
CLARIFY

иначе criteria insufficient information
    ↓
MENTOR_REVIEW

иначе fully ready and low/no risks
    ↓
ACCEPT

иначе
    ↓
MENTOR_REVIEW
```

### Операторы conditions

Arbiter поддерживает операторы:

- `exists`;
- `not_exists`;
- `eq`;
- `ne`;
- `gt`;
- `gte`;
- `lt`;
- `lte`;
- `in`;
- `not_in`;
- `contains`;
- `any_in`;
- `all_in`.

Для числовых сравнений `_compare_numbers()` возвращает `False`, если значения не int/float. Это защищает от конфигурационной ошибки, когда, например, строку пытаются сравнить оператором `gt`.

### Где здесь используется LLM

Этот этап детерминированный: LLM здесь не вызывается.

# Генерация уточняющих вопросов (`app/pipeline/question_generator.py`)

Этот файл отвечает за вопросы заказчику. Он не вызывает LLM, а использует шаблоны из `config/question_templates.json`.

```text
AIContext
└── completeness_result
    └── missing_information
          │
          ▼
TemplateQuestionGeneratorStage.generate()
          │
          ├── для каждого missing item
          ├── взять template по item.field_key
          └── создать ClarificationQuestion
          │
          ▼
QuestionGenerationResult
          │
          ▼
AIContext.with_clarification_result()
```

## Импорты

`json` нужен для чтения `question_templates.json`.

`Path` нужен для пути к файлу шаблонов.

`CompletenessResult`, `ClarificationQuestion`, `QuestionGenerationResult` - модели входа и результата.

`AssessmentRecommendation` добавляется только в summary, если assessment result есть.

## Класс `TemplateQuestionGeneratorStage`

### Что это

Детерминированный stage, который превращает missing fields в вопросы.

### Зачем он нужен

Вопросы по отсутствующим обязательным полям лучше держать стабильными и редактируемыми в конфиге. Если missing field `project_goal`, вопрос всегда один и тот же: "Какую основную цель должен решить проект...".

### Что получает

`AIContext` с `completeness_result`.

### Что делает

1. Загружает templates из JSON.
2. Идёт по `completeness_result.missing_information`.
3. Для каждого missing field ищет шаблон по `field_key`.
4. Если шаблон есть, создаёт `ClarificationQuestion`.
5. Если шаблона нет, записывает field key в `missing_template_fields`.
6. Возвращает `QuestionGenerationResult`.

Этот stage генерирует вопросы только для `missing_information`, а не для `clarification_information`.

### Что возвращает

```text
QuestionGenerationResult
├── questions: list[ClarificationQuestion]
├── summary: str | None
└── technical_info
    ├── llm_invoked: false
    ├── question_count
    └── missing_template_fields
```

### Где здесь используется LLM

Этот этап детерминированный: LLM здесь не вызывается.

# MVP-планирование (`app/pipeline/mvp_planner.py`)

Этот файл отвечает за условный LLM-вызов для упрощения проекта до MVP.

```text
AIContext
├── brief_input
├── extracted_brief
├── assessment_result
└── arbitration_result
          │
          ▼
MVPPlannerStage.plan_context()
          │
          ▼
plan_assessment()
          │
          ├── if arbitration.final_status != SIMPLIFY
          │       └── MVPPlanningResult(plan=None, llm_invoked=false)
          │
          └── if arbitration.final_status == SIMPLIFY
                  │
                  ▼
              prompts/mvp_planner.md
                  │
                  ▼
              LLMRunner.run_json(response_model=MVPPlan)
                  │
                  ▼
              MVPPlanningResult(plan=MVPPlan, llm_invoked=true)
```

## Класс `MVPPlannerStage`

### Что это

LLM-stage, который работает только при статусе `SIMPLIFY`.

### Зачем он нужен

Если arbiter решил, что проект слишком широкий или рискованный по scope, заказчику нужен не просто статус "упростить", а конкретное предложение: что оставить, что убрать, как выглядит первая версия.

### Что получает

`AIContext` с:

- `brief_input`;
- `extracted_brief`;
- `assessment_result`;
- `arbitration_result`.

Если `extracted_brief`, `assessment_result` или `arbitration_result` отсутствуют, stage выбрасывает `MVPPlannerError`.

### Что делает

Если `arbitration_result.final_status != DecisionStatus.simplify`, LLM не вызывается. Возвращается `MVPPlanningResult` с:

- `plan=None`;
- `technical_info.llm_invoked=False`;
- `skipped_reason="MVP planner runs only when arbitration status is SIMPLIFY"`.

Если статус `SIMPLIFY`:

1. собирает JSON-контекст из brief, extraction, assessment и arbitration;
2. рендерит `prompts/mvp_planner.md`;
3. вызывает `_execute_structured_stage()`;
4. ожидает `MVPPlan`;
5. проверяет `_validate_plan()`.

`_validate_plan()` требует, чтобы `keep`, `simplify`, `mvp_scope`, `rationale` не были пустыми. Это бизнес-валидация поверх Pydantic-схемы.

## Модели MVP (`app/schemas/mvp.py`)

```text
MVPPlan
├── core_goal: str
├── keep: list[str]
├── remove: list[str]
├── simplify: list[str]
├── mvp_scope: list[str]
└── rationale: list[str]

MVPPlanningResult
├── plan: MVPPlan | None
└── technical_info: MVPPlanningTechnicalInfo
```

### Где здесь используется LLM

LLM вызывается только если `arbitration_result.final_status == DecisionStatus.simplify`.

Prompt: `prompts/mvp_planner.md`.

Данные, передаваемые модели: JSON с `brief_input`, `extracted_brief`, `risk_analysis_result`, `evaluation_result`, `arbitration_result`.

Ожидаемый результат: JSON по схеме `MVPPlan`.

Что происходит дальше: `ResponseWriterStage` использует plan для текста ответа, а `BriefAnalysisResultBuilder` использует его для `mvp_suggestion`.

# Финальный ответ (`app/pipeline/response_writer.py`, `app/pipeline/result_builder.py`)

Эти файлы отвечают за финальный результат. Они не вызывают LLM.

```text
AIContext
├── extracted_brief
├── completeness_result
├── assessment_result
├── arbitration_result
├── clarification_result
└── mvp_planning_result
          │
          ▼
ResponseWriterStage.write_context()
          │
          ├── _build_response_text()
          │   ├── ACCEPT -> _accept_response()
          │   ├── CLARIFY -> _clarify_response()
          │   ├── SIMPLIFY -> _simplify_response()
          │   ├── MENTOR_REVIEW -> _mentor_review_response()
          │   └── REJECT -> _reject_response()
          │
          ├── context.with_final_response(text)
          │
          ├── BriefAnalysisResultBuilder.build()
          │
          └── context.with_final_response(text, payload)
          │
          ▼
BriefAnalysisResult
```

## `ResponseWriterStage`

### Что это

Детерминированный stage, который пишет черновик ответа заказчику по финальному статусу.

### Зачем он нужен

Итоговый текст должен быть стабильным и соответствовать бизнес-статусу. LLM уже поучаствовала в извлечении и анализе, но финальное письмо собирается из проверенных структурированных данных.

### Что получает

`AIContext` с `arbitration_result`. Для некоторых статусов дополнительно используются questions, MVP и reasons.

### Что делает

`_build_response_text()` выбирает шаблон по `DecisionStatus`:

- `ACCEPT` -> благодарность и следующий шаг;
- `CLARIFY` -> список уточняющих вопросов;
- `SIMPLIFY` -> MVP-предложение и optional questions;
- `MENTOR_REVIEW` -> сообщение о необходимости экспертизы наставника;
- `REJECT` -> объяснение, что проект не подходит в текущем виде.

Если статус неизвестен, выбрасывается `ResponseWriterError`.

`_summary()` берёт summary из assessment, если он есть. Если нет, использует `extracted_brief.project_goal.value`. Если и его нет, пишет fallback.

`_format_questions()` возвращает вопросы из `clarification_result`. Если вопросов нет и `optional=False`, возвращается текст "Уточняющие вопросы пока не сформированы."

## `BriefAnalysisResultBuilder`

### Что это

Класс, который превращает полный `AIContext` в публичную модель `BriefAnalysisResult`.

### Зачем он нужен

Внутренний `AIContext` содержит много технических деталей: raw responses, trace info, конфиг, промежуточные модели. Наружу CLI должен отдать компактный JSON с итогом.

### Что получает

`AIContext` с обязательными данными:

- `extracted_brief`;
- `completeness_result`;
- `assessment_result`;
- `arbitration_result`;
- `final_response_text`.

### Что проверяет

`_validate_context()` выбрасывает `BriefAnalysisResultError`, если не хватает обязательных частей.

Дополнительные правила:

- если final status `CLARIFY`, должны быть clarification questions;
- если final status `SIMPLIFY`, должен быть MVP plan.

Это значит, что pipeline не должен вернуть статус `SIMPLIFY` без фактического плана. В текущей архитектуре это обеспечивается условным LLM-вызовом `MVPPlannerStage`.

### Что возвращает

```text
BriefAnalysisResult
├── summary
├── extracted_fields
│   ├── goal
│   ├── expected_result
│   ├── tasks
│   ├── domain
│   ├── direction
│   ├── available_materials
│   ├── missing_information
│   └── complexity_factors
├── assessment
│   ├── recommendation
│   ├── confidence
│   ├── reasons
│   └── risks
├── clarifying_questions
├── mvp_suggestion
└── customer_response_draft
```

`_STATUS_MAP` переводит внутренний enum `DecisionStatus.accept` в строку `"accept"` для публичного JSON.

`_confidence_label()` превращает float confidence в `"low"`, `"medium"` или `"high"`:

- `< 0.45` -> low;
- `< 0.75` -> medium;
- иначе high;
- если confidence нет -> medium.

`_deduplicate()` удаляет повторы, сохраняя порядок. Для этого используется `seen: set[str]`.

### Где здесь используется LLM

Этот этап детерминированный: LLM здесь не вызывается.

# Финальные схемы результата (`app/schemas/final_result.py`)

Это модели публичного JSON, который печатает CLI.

```text
BriefAnalysisResult
├── summary: str
├── extracted_fields: BriefExtractedFields
├── assessment: BriefAssessmentSummary
├── clarifying_questions: list[str]
├── mvp_suggestion: str
└── customer_response_draft: str
```

`RecommendationValue` и `ConfidenceValue` объявлены через `Literal`. Это type hint, который ограничивает допустимые строковые значения. Например, `recommendation` не может быть произвольной строкой, только `"accept"`, `"clarify"`, `"simplify"`, `"mentor_review"` или `"reject"`.

`BriefExtractedFields` - упрощённая версия `ExtractedBrief` для внешнего результата. Внутри pipeline факты хранятся богаче: со статусами, evidence и notes. Наружу отдаётся только то, что нужно пользователю результата.

# Конфигурация критериев (`app/config/criteria.py`, `config/criteria.yaml`)

`criteria.yaml` хранит бизнес-конфигурацию:

- типы проектов;
- типы задач;
- критерии оценки;
- обязательные поля;
- типы рисков;
- правила арбитража.

`app/config/criteria.py` описывает Pydantic-модели для этой конфигурации и загрузчик.

```text
config/criteria.yaml
     │
     ▼
CriteriaLoader.load()
     │
     ├── read_text(utf-8)
     ├── _load_yaml_like()
     ├── CriteriaConfig.model_validate()
     └── CriteriaConfig
          │
          ├── CompletenessCheckStage
          ├── AssessmentPreparation / AssessmentStage
          └── DeterministicArbiterStage
```

Важно: в проекте не используется PyYAML. Файл парсится собственным простым YAML-like парсером: `_preprocess_yaml_lines()`, `_parse_block()`, `_parse_mapping()`, `_parse_list()`, `_parse_scalar()`.

Назначение такого парсера по коду можно определить только фактически: он поддерживает ограниченное подмножество YAML, достаточное для текущего `criteria.yaml`. Почему автор выбрал не PyYAML, из кода однозначно не следует.

`@lru_cache(maxsize=1)` у `get_criteria_config()` кэширует результат загрузки. Первый вызов читает файл, следующие возвращают уже загруженный объект.

## Основные модели конфигурации

```text
CriteriaConfig
└── evaluation: EvaluationConfiguration
    ├── version
    ├── description
    ├── project_types: list[ProjectType]
    ├── task_types: list[TaskType]
    ├── criteria: list[Criterion]
    ├── required_fields: list[RequiredField]
    ├── decision_thresholds
    ├── risk_analysis: RiskAnalysisConfiguration | None
    └── arbitration: ArbitrationConfiguration | None
```

`CompletenessCheckStage` использует `required_fields`.

`AssessmentStage` использует `criteria` и `risk_analysis.risk_types`.

`DeterministicArbiterStage` использует `arbitration.rules`.

# Настройки приложения (`app/config/settings.py`)

Этот файл загружает `.env` и описывает runtime-настройки.

```text
.env
     │
     ▼
Config.load()
     │
     ├── load_dotenv()
     ├── Settings(_env_file=...)
     └── Settings
          ├── openrouter_api_key
          ├── openrouter_model
          ├── langfuse_*
          ├── debug
          ├── log_level
          └── knowledge_*
```

`pydantic_settings.BaseSettings` умеет брать значения из переменных окружения и `.env`.

`Field(..., alias="OPENROUTER_API_KEY")` означает, что поле обязательно и читается из переменной `OPENROUTER_API_KEY`.

`Literal["DEBUG", "INFO", ...]` ограничивает допустимые уровни логирования.

`@model_validator(mode="after")` проверяет связь настроек chunking: `knowledge_chunk_overlap` должен быть меньше `knowledge_chunk_size`.

Секреты из `.env` в документации не раскрываются.

# Эволюция данных через pipeline

```text
сырой текст / файл
    ↓
BriefInput
    ├── original_text
    ├── normalized_text
    └── metadata
    ↓
AIContext
    └── inputs.brief_input
    ↓
ExtractionResult
    └── ExtractedBrief
        ├── project_goal: ExtractedFact
        ├── tasks: list[ExtractedFact]
        ├── project_direction: ExtractedFact
        └── ...
    ↓
CompletenessResult
    ├── present_information
    ├── missing_information
    └── clarification_information
    ↓
AssessmentResult
    ├── criterion_evaluations
    ├── risks
    └── assessment recommendation
    ↓
ArbitrationResult
    └── final_status
    ↓
QuestionGenerationResult
    └── questions
    ↓
MVPPlanningResult
    ├── plan=None, если не SIMPLIFY
    └── plan=MVPPlan, если SIMPLIFY
    ↓
ResponseState
    ├── final_response_text
    └── final_response_payload
    ↓
BriefAnalysisResult
```

# Где применяется LLM, а где обычный Python

```text
BriefInputFactory                  [PYTHON]
BriefInputNormalizer               [PYTHON]
AIContext.from_brief               [PYTHON]
Extractor                          [LLM]
CompletenessCheckStage             [PYTHON]
AssessmentPreparation              [PYTHON]
AssessmentStage                    [LLM]
DeterministicArbiterStage          [PYTHON]
TemplateQuestionGeneratorStage     [PYTHON]
MVPPlannerStage                    [CONDITIONAL LLM]
ResponseWriterStage                [PYTHON]
BriefAnalysisResultBuilder         [PYTHON]
```

LLM-вызовы:

1. `Extractor`
   - prompt: `prompts/extractor.md`;
   - input: normalized brief text;
   - output model: `ExtractedBrief`.

2. `AssessmentStage`
   - prompt: `prompts/assessment.md`;
   - input: normalized brief, extracted brief, completeness result, criteria, risk types, optional retrieved context;
   - output model: `AssessmentPayload`.

3. `MVPPlannerStage`
   - prompt: `prompts/mvp_planner.md`;
   - input: brief, extracted brief, assessment result, arbitration result;
   - output model: `MVPPlan`;
   - вызывается только при `final_status == SIMPLIFY`.

# Почему архитектура разделена так

Разделение по этапам делает pipeline понятным и проверяемым.

`Extractor` отделён от `AssessmentStage`, потому что извлечение фактов и оценка качества - разные задачи. Если их смешать, LLM может начать делать выводы там, где нужны только факты.

`CompletenessCheckStage` отделён от LLM, потому что полнота по обязательным полям - это правило, а не творческая оценка. Код может прозрачно объяснить, какое поле отсутствует.

`AssessmentStage` использует LLM, потому что риски и критерии часто требуют интерпретации текста.

`DeterministicArbiterStage` отделён от LLM, потому что финальное решение должно быть воспроизводимым. LLM даёт аналитические сигналы, но не управляет итоговым статусом напрямую.

`TemplateQuestionGeneratorStage` не использует LLM, потому что вопросы по missing fields можно стабильно хранить в JSON-шаблонах.

`MVPPlannerStage` условный, потому что MVP-план нужен только для статуса `SIMPLIFY`. Для остальных статусов LLM-вызов был бы лишней стоимостью и лишним источником нестабильности.

`ResponseWriterStage` детерминированный, потому что финальная коммуникация должна строго соответствовать решению arbiter-а.

# Компоненты вне основного production pipeline

В репозитории есть дополнительные модули, которые экспортируются или тестируются, но не входят в список stages, создаваемый `BriefAnalysisPipeline.from_llm_client()`:

- `app/pipeline/self_check.py`;
- часть `app/knowledge/*`;
- security-модули `app/security/*`.

Они могут быть полезны для расширения системы, тестов или будущих сценариев, но в текущем production pipeline одного брифа через `app/main.py` они не запускаются. Поэтому в основной схеме выше они не показаны как обязательные этапы.

# Короткая карта файлов

```text
app/main.py
└── CLI: аргументы, загрузка настроек, запуск pipeline

app/input/brief_input.py
└── чтение и нормализация входного текста

app/schemas/*.py
└── Pydantic-модели данных между этапами

app/pipeline/orchestrator.py
└── сборка и последовательный запуск stages

app/pipeline/contracts.py
└── общий контракт BaseStage

app/pipeline/base.py
└── общий каркас LLM-stage

app/pipeline/extractor.py
└── LLM-извлечение ExtractedBrief

app/pipeline/completeness.py
└── Python-проверка обязательных полей

app/pipeline/assessment.py
└── LLM-оценка критериев и рисков

app/pipeline/arbiter.py
└── Python-правила финального решения

app/pipeline/question_generator.py
└── Python-генерация вопросов из JSON-шаблонов

app/pipeline/mvp_planner.py
└── условный LLM-план MVP

app/pipeline/response_writer.py
└── Python-сборка текста ответа заказчику

app/pipeline/result_builder.py
└── Python-сборка публичного BriefAnalysisResult

app/llm/*.py
└── интерфейс LLM, OpenRouter-клиент, runner с JSON Schema и retries

app/prompts/manager.py
└── загрузка и рендер prompt-файлов

config/criteria.yaml
└── критерии, required fields, risk types, arbitration rules

config/question_templates.json
└── шаблоны уточняющих вопросов
```
