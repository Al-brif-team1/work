import unittest

from pydantic import ValidationError

from app.schemas import (
    AssessmentPayload,
    AssessmentRecommendation,
    AssessmentResult,
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
)


class TestTrafficLightSchema(unittest.TestCase):
    def test_traffic_light_result_validates_status_and_strips_text(self) -> None:
        result = TrafficLightResult(
            status="green",
            direction=" Программирование ",
            specialization=" Питон/питон+ ",
            reason=" задача входит в навыки студентов ",
            matches=[
                TrafficLightMatch(
                    task=" создание телеграм-бота ",
                    matched_rule=" создание телеграм-бота ",
                    status="green",
                    reason=" задача входит в навыки студентов ",
                )
            ],
        )

        self.assertEqual(result.status, TrafficLightStatus.green)
        self.assertEqual(result.direction, "Программирование")
        self.assertEqual(result.specialization, "Питон/питон+")
        self.assertEqual(result.matches[0].task, "создание телеграм-бота")
        self.assertEqual(result.matches[0].matched_rule, "создание телеграм-бота")
        self.assertEqual(result.matches[0].status, TrafficLightStatus.green)
        self.assertEqual(result.reason, "задача входит в навыки студентов")

    def test_unknown_result_accepts_empty_matches(self) -> None:
        result = TrafficLightResult()

        self.assertEqual(result.status, TrafficLightStatus.unknown)
        self.assertEqual(result.matches, [])

    def test_multiple_matches_with_different_statuses_validate(self) -> None:
        result = TrafficLightResult(
            status="yellow",
            direction="Программирование",
            specialization="Фулстек-разработчик",
            matches=[
                {
                    "task": "сайт заметок с авторизацией",
                    "matched_rule": "сайт заметок с авторизацией",
                    "status": "green",
                    "reason": "задача входит в навыки студентов",
                },
                {
                    "task": "комплексная AI/ML интеграция",
                    "matched_rule": "комплексные AI/ML интеграции",
                    "status": "red",
                    "reason": "задача не входит в навыки студентов",
                },
            ],
            reason="есть задачи разных цветов",
        )

        self.assertEqual(len(result.matches), 2)
        self.assertEqual(result.matches[0].status, TrafficLightStatus.green)
        self.assertEqual(result.matches[1].status, TrafficLightStatus.red)

    def test_traffic_light_result_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            TrafficLightResult(status="yellow", extra_field="not allowed")

        with self.assertRaises(ValidationError):
            TrafficLightMatch(
                task="task",
                matched_rule="rule",
                status="green",
                reason="reason",
                extra_field="not allowed",
            )

    def test_traffic_light_result_rejects_invalid_status(self) -> None:
        with self.assertRaises(ValidationError):
            TrafficLightResult(status="blue")

    def test_assessment_payload_defaults_traffic_light_to_unknown(self) -> None:
        payload = AssessmentPayload(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
        )

        self.assertEqual(payload.traffic_light.status, TrafficLightStatus.unknown)

    def test_assessment_result_defaults_traffic_light_to_unknown(self) -> None:
        result = AssessmentResult(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.unknown)


if __name__ == "__main__":
    unittest.main()
