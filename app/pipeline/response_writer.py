"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml

from app.pipeline.contracts import BaseStage
from app.pipeline.result_builder import (
    BriefAnalysisResultBuilder,
    build_public_decision_summary,
    deduplicate,
)
from app.schemas import (
    AIContext,
    CriterionEvaluationStatus,
    DecisionStatus,
    RiskSeverity,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient


class ResponseWriterError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class _RejectLetters(NamedTuple):
    """Разобранный config/reject_letters.yaml. Основание отказа либо заменяет письмо целиком (letters), либо только добавляет свою причину к концовке (closing_prefixes)."""

    letters: dict[str, str]
    closing_prefixes: dict[str, str]
    default_closing: str


class ResponseWriterStage(BaseStage[AIContext, AIContext]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Секретарь-Писатель. Он детерминированным кодом собирает готовый ответ из деталей конструктора. Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    # Согласие обосновывают выполненные критерии, отказ и эскалацию - проваленные.
    # Из рисков показываем только высокие и критические: остальные не влияют на решение.
    _ACCEPTED_CRITERION_STATUSES = frozenset({CriterionEvaluationStatus.met})
    _PROBLEM_CRITERION_STATUSES = frozenset(
        {
            CriterionEvaluationStatus.not_met,
            CriterionEvaluationStatus.insufficient_information,
            CriterionEvaluationStatus.risk_detected,
        }
    )
    _REPORTED_RISK_SEVERITIES = frozenset({RiskSeverity.high, RiskSeverity.critical})

    # Отказы различаются не строгостью, а основанием, и заказчик должен понять, что
    # именно ему делать дальше. Основание опознается по типу риска, а порядок задает
    # приоритет: запрет темы сильнее несоответствия формата, тот - сильнее
    # запредельной ответственности.
    _REJECT_GROUND_PRIORITY = (
        "restricted_topic",
        "out_of_scope_request",
        "production_criticality",
    )
    # Сами формулировки писем лежат в config/reject_letters.yaml: их правит тот, кто
    # отвечает за переписку с заказчиком, и ради запятой трогать код не нужно.

    def __init__(
        self,
        *,
        result_builder: BriefAnalysisResultBuilder | None = None,
        reject_letters_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        self._result_builder = result_builder or BriefAnalysisResultBuilder()
        self._reject_letters = self._load_reject_letters(
            reject_letters_path or self._default_reject_letters_path()
        )

    def write_context(self, context: AIContext) -> AIContext:
        """Выполняет шаг «write context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        response_text = self._build_response_text(context)
        context_with_text = context.with_final_response(response_text)
        payload = self._result_builder.build(context_with_text)
        return context_with_text.with_final_response(
            response_text,
            response_payload=payload.model_dump(mode="json"),
        )

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
        return self.write_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return exc

    def _build_trace_input(self, stage_input: AIContext) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "has_arbitration_result": stage_input.arbitration_result is not None,
            "has_questions": bool(
                stage_input.clarification_result
                and stage_input.clarification_result.questions
            ),
            "has_mvp_plan": bool(
                stage_input.mvp_planning_result
                and stage_input.mvp_planning_result.plan
            ),
        }

    def _build_trace_output(self, stage_output: AIContext) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "status": "success",
            "response_length": len(stage_output.final_response_text or ""),
        }

    def _build_response_text(self, context: AIContext) -> str:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        if context.arbitration_result is None:
            raise ResponseWriterError("Response writer requires arbitration_result")

        status = context.arbitration_result.final_status
        if status is DecisionStatus.accept:
            return self._accept_response(context)
        if status is DecisionStatus.clarify:
            return self._clarify_response(context)
        if status is DecisionStatus.simplify:
            return self._simplify_response(context)
        if status is DecisionStatus.mentor_review:
            return self._mentor_review_response(context)
        if status is DecisionStatus.accept_with_clarifications:
            return self._accept_with_clarifications_response(context)
        if status is DecisionStatus.reject:
            return self._reject_response(context)

        raise ResponseWriterError(f"Unsupported arbitration status: {status}")

    def _accept_response(self, context: AIContext) -> str:
        """Выполняет шаг «accept response». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        summary = self._summary(context)
        reasons = self._format_reasons(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке проект можно брать "
            "в работу как студенческий проект.\n\n"
            f"Кратко о проекте: {summary}\n\n"
            f"{reasons}"
            "Следующий шаг: согласовать состав работ, сроки и формат материалов "
            "для старта команды."
        )

    def _clarify_response(self, context: AIContext) -> str:
        """Выполняет шаг «clarify response». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        questions = self._format_questions(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. Сейчас по заявке не хватает данных для корректной "
            "оценки студенческого проекта.\n\n"
            f"{questions}\n\n"
            "После уточнений мы сможем повторно оценить реалистичность проекта, "
            "риски и формат участия студентов."
        )

    def _accept_with_clarifications_response(self, context: AIContext) -> str:
        """Выполняет шаг «accept with clarifications response». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        summary = self._summary(context)
        reasons = self._format_reasons(context)
        questions = self._format_questions(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке проект можно брать "
            "в работу как студенческий проект, но перед стартом нужно уточнить "
            "несколько деталей.\n\n"
            f"Кратко о проекте: {summary}\n\n"
            f"{reasons}"
            f"Уточняющие вопросы:\n{questions}\n\n"
            "Следующий шаг: согласовать состав работ, сроки и формат материалов "
            "для старта команды."
        )

    def _simplify_response(self, context: AIContext) -> str:
        """Выполняет шаг «simplify response». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        mvp = self._format_mvp(context)
        questions = self._format_questions(context, optional=True)
        question_block = f"\n\nДополнительно нужно уточнить:\n{questions}" if questions else ""
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. Идея потенциально подходит для студенческого "
            "проекта, но текущий объём выглядит слишком широким для первой "
            f"версии.\n\n{mvp}{question_block}\n\n"
            "Предлагаем обсудить такой сокращённый формат и отдельно вынести "
            "промышленные или сильно рискованные задачи за рамки MVP."
        )

    def _mentor_review_response(self, context: AIContext) -> str:
        """Выполняет шаг «mentor review response». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        reasons = self._format_reasons(context)
        questions = self._format_questions(context, optional=True)
        question_block = f"\n\nВопросы для уточнения:\n{questions}" if questions else ""
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке заявка требует "
            "дополнительной экспертизы наставника перед финальным решением.\n\n"
            f"{reasons}{question_block}\n\n"
            "Мы вернёмся с рекомендацией после экспертной проверки состава работ "
            "и рисков."
        )

    def _reject_response(self, context: AIContext) -> str:
        """Собирает отказное письмо под основание отказа. Запрещенной теме и заявке не о заказе работы отвечаем одной фразой: перечень оснований и приглашение сократить объем там только вводят заказчика в заблуждение."""
        assessment = context.assessment_result
        # Порог тот же, что и у блока оснований: риск, который заказчику даже
        # не показывают, не должен определять текст письма.
        reported_types = (
            {
                risk.type
                for risk in assessment.risks
                if risk.severity in self._REPORTED_RISK_SEVERITIES
            }
            if assessment is not None
            else set()
        )
        ground = next(
            (item for item in self._REJECT_GROUND_PRIORITY if item in reported_types),
            None,
        )

        letter = self._reject_letters.letters.get(ground)
        if letter is not None:
            return letter

        reasons = self._format_reasons(context)
        closing_prefix = self._reject_letters.closing_prefixes.get(ground)
        default_closing = self._reject_letters.default_closing
        closing = (
            f"{closing_prefix} {default_closing}"
            if closing_prefix is not None
            else default_closing
        )
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке проект в текущем виде "
            "не подходит для формата студенческой работы.\n\n"
            f"{reasons}{closing}"
        )

    def _summary(self, context: AIContext) -> str:
        """Выполняет шаг «summary». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if context.arbitration_result is not None:
            return build_public_decision_summary(context)
        if context.assessment_result and context.assessment_result.summary:
            return context.assessment_result.summary
        if context.extracted_brief and context.extracted_brief.project_goal.value:
            return context.extracted_brief.project_goal.value
        return "описание требует дополнительного уточнения"

    def _format_reasons(self, context: AIContext) -> str:
        """Собирает блок «Основания оценки» из объяснений критериев. Причины арбитра сюда не идут: это технические строки конфигурации, а заказчику нужна оценка его брифа."""
        reasons = deduplicate(self._reason_items(context))
        if not reasons:
            return ""
        return "Основания оценки:\n" + "\n".join(f"- {item}" for item in reasons) + "\n\n"

    def _reason_items(self, context: AIContext) -> list[str]:
        """Отбирает объяснения под конкретный вердикт. Без отбора отказное письмо начиналось бы с похвал брифу, а настоящая причина отказа терялась бы в конце списка."""
        assessment = context.assessment_result
        if assessment is None:
            return []

        accepted = (
            context.arbitration_result is not None
            and context.arbitration_result.final_status
            in {
                DecisionStatus.accept,
                DecisionStatus.accept_with_clarifications,
            }
        )
        reported_statuses = (
            self._ACCEPTED_CRITERION_STATUSES
            if accepted
            else self._PROBLEM_CRITERION_STATUSES
        )
        items = [
            item.explanation
            for item in assessment.criterion_evaluations
            if item.explanation and item.status in reported_statuses
        ]
        items.extend(
            risk.description
            for risk in assessment.risks
            if risk.severity in self._REPORTED_RISK_SEVERITIES
        )
        return items

    def _format_questions(self, context: AIContext, *, optional: bool = False) -> str:
        """Выполняет шаг «format questions». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        questions = (
            context.clarification_result.questions
            if context.clarification_result is not None
            else []
        )
        if not questions:
            return "" if optional else "Уточняющие вопросы пока не сформированы."
        return "\n".join(f"- {item.question}" for item in questions)

    def _format_mvp(self, context: AIContext) -> str:
        """Выполняет шаг «format mvp». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        planning = context.mvp_planning_result
        if planning is None or planning.plan is None:
            return (
                "MVP-предложение пока не сформировано: требуется дополнительная "
                "проработка объёма."
            )

        plan = planning.plan
        parts = [f"Предлагаемый MVP: {plan.core_goal}"]
        if plan.keep:
            parts.append("Оставить в первой версии: " + "; ".join(plan.keep))
        if plan.simplify:
            parts.append("Упростить: " + "; ".join(plan.simplify))
        if plan.remove:
            parts.append("Исключить из первой версии: " + "; ".join(plan.remove))
        return "\n".join(parts)

    @staticmethod
    def _load_reject_letters(path: str | Path) -> _RejectLetters:
        """Читает формулировки отказных писем из YAML-ресурса. Текст не нормализуем: заказчику уходит ровно то, что записано в файле."""
        section = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["reject_letters"]
        grounds = section["grounds"]
        return _RejectLetters(
            letters={
                item["key"]: item["letter"] for item in grounds if "letter" in item
            },
            closing_prefixes={
                item["key"]: item["closing_prefix"]
                for item in grounds
                if "closing_prefix" in item
            },
            default_closing=section["default_closing"],
        )

    @staticmethod
    def _default_reject_letters_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "config" / "reject_letters.yaml"
