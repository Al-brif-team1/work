from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from app.main import (
    TrafficLightDiagnosticsStage,
    add_traffic_light_diagnostics_stage,
    print_traffic_light_diagnostics,
)
from app.pipeline import AssessmentStage
from app.schemas import (
    AssessmentRecommendation,
    AssessmentResult,
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
)


class TestCliTrafficLightDiagnostics(unittest.TestCase):
    def test_inserts_diagnostics_stage_after_assessment_stage(self) -> None:
        class PipelineStub:
            def __init__(self) -> None:
                self.insert_calls = []

            def insert_stage_after(self, stage_type, stage) -> bool:
                self.insert_calls.append((stage_type, stage))
                return True

        pipeline = PipelineStub()

        add_traffic_light_diagnostics_stage(pipeline)

        self.assertEqual(len(pipeline.insert_calls), 1)
        stage_type, stage = pipeline.insert_calls[0]
        self.assertIs(stage_type, AssessmentStage)
        self.assertIsInstance(stage, TrafficLightDiagnosticsStage)

    def test_prints_traffic_light_diagnostics_with_matches(self) -> None:
        result = AssessmentResult(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
            traffic_light=TrafficLightResult(
                status=TrafficLightStatus.green,
                direction="Programming",
                specialization="Python",
                matches=[
                    TrafficLightMatch(
                        task="Build a Telegram bot",
                        matched_rule="simple bot",
                        status=TrafficLightStatus.green,
                        reason="Fits student task scope.",
                    )
                ],
            ),
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            print_traffic_light_diagnostics(result)

        self.assertEqual(
            stderr.getvalue(),
            "\n[TRAFFIC LIGHT DIAGNOSTICS]\n"
            "status=green\n"
            "direction=Programming\n"
            "specialization=Python\n"
            "matches:\n"
            "  - task=Build a Telegram bot\n"
            "    matched_rule=simple bot\n"
            "    status=green\n"
            "    reason=Fits student task scope.\n",
        )

    def test_prints_empty_matches(self) -> None:
        result = AssessmentResult(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
            traffic_light=TrafficLightResult(),
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            print_traffic_light_diagnostics(result)

        self.assertEqual(
            stderr.getvalue(),
            "\n[TRAFFIC LIGHT DIAGNOSTICS]\n"
            "status=unknown\n"
            "direction=None\n"
            "specialization=None\n"
            "matches: []\n",
        )


if __name__ == "__main__":
    unittest.main()
