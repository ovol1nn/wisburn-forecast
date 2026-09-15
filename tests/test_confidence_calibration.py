from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "confidence" / "calibrate_forecast_confidence.py"
SPEC = importlib.util.spec_from_file_location("forecast_confidence", MODULE)
assert SPEC and SPEC.loader
confidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confidence
SPEC.loader.exec_module(confidence)


class ForecastConfidenceTests(unittest.TestCase):
    def test_score_rewards_skill_direction_and_calibration(self) -> None:
        strong = confidence.confidence_score(0.60, 0.90, 0.90)
        weak = confidence.confidence_score(-0.10, 0.55, 0.50)
        self.assertGreater(strong, weak)
        self.assertGreaterEqual(strong, 0.75)
        self.assertEqual(confidence.confidence_level(strong, 0.60, 0.90), "high")

    def test_negative_skill_is_low_even_when_interval_is_calibrated(self) -> None:
        score = confidence.confidence_score(-0.01, 0.85, 0.90)
        self.assertEqual(confidence.confidence_level(score, -0.01, 0.85), "low")

    def test_horizons_are_covered_by_current_contracts(self) -> None:
        self.assertEqual(confidence.HORIZONS_SECONDS[-1], 600)
        self.assertEqual(len(confidence.SPECS), 2)


if __name__ == "__main__":
    unittest.main()
