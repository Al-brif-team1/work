import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import (
    TrafficLightConfigError,
    TrafficLightConfiguration,
    TrafficLightLoader,
    get_traffic_light_config,
)


class TestTrafficLightConfig(unittest.TestCase):
    def setUp(self) -> None:
        get_traffic_light_config.cache_clear()

    def tearDown(self) -> None:
        get_traffic_light_config.cache_clear()

    def test_loads_default_traffic_light_config(self) -> None:
        config = TrafficLightLoader.load()

        self.assertIsInstance(config.traffic_light, TrafficLightConfiguration)
        direction_titles = [item.title for item in config.traffic_light.directions]
        self.assertIn("Программирование", direction_titles)
        self.assertIn("Дизайн", direction_titles)

        python = config.traffic_light.directions[0].specializations[0]
        self.assertEqual(python.title, "Питон/питон+")
        self.assertIn("создание телеграм-бота", python.green)
        self.assertIn("разработка игр на unity", python.red)

    def test_get_traffic_light_config_is_cached(self) -> None:
        sentinel = TrafficLightLoader.load()

        with patch.object(TrafficLightLoader, "load", return_value=sentinel) as mocked:
            first = get_traffic_light_config()
            second = get_traffic_light_config()

        self.assertIs(first, second)
        self.assertEqual(mocked.call_count, 1)

    def test_invalid_yaml_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.yaml"
            path.write_text(
                "traffic_light:\n  version: [broken\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TrafficLightConfigError,
                "Invalid YAML syntax in traffic light configuration",
            ):
                TrafficLightLoader.load(path)

    def test_invalid_schema_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.yaml"
            path.write_text("traffic_light:\n  version: '1'\n", encoding="utf-8")

            with self.assertRaisesRegex(
                TrafficLightConfigError,
                "Invalid traffic light configuration schema",
            ):
                TrafficLightLoader.load(path)


if __name__ == "__main__":
    unittest.main()
