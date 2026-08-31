from __future__ import annotations

import json
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODULE_ROOT.parents[1]


class ForecastModuleLayoutTests(unittest.TestCase):
    def test_contract_has_two_current_models(self) -> None:
        contract = json.loads((MODULE_ROOT / "contracts" / "current_contracts.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["models"]["model1"]["seq_len"], 120)
        self.assertEqual(contract["models"]["model1"]["pred_len"], 60)
        self.assertEqual(contract["models"]["model1"]["native_step_seconds"], 10)
        self.assertEqual(contract["models"]["model2"]["seq_len"], 1200)
        self.assertEqual(contract["models"]["model2"]["pred_len"], 600)
        self.assertEqual(contract["models"]["model2"]["native_step_seconds"], 1)
        self.assertEqual(len(contract["models"]["model1"]["target_columns"]), 5)
        self.assertEqual(len(contract["models"]["model2"]["target_columns"]), 5)

    def test_source_module_has_no_generated_training_output(self) -> None:
        self.assertFalse((MODULE_ROOT / "training_ready").exists())
        self.assertFalse((MODULE_ROOT / "training" / "outputs").exists())
        self.assertTrue((MODULE_ROOT / "data_engineering" / "build_model_datasets.py").is_file())
        self.assertTrue((MODULE_ROOT / "data_engineering" / "prepare_dataset_splits.py").is_file())
        self.assertTrue((MODULE_ROOT / "tslib" / "run.py").is_file())
        self.assertTrue((WORKSPACE_ROOT / "offline" / "artifacts" / "forecast").is_dir())

    def test_training_paths_use_workspace_or_explicit_environment(self) -> None:
        data_builder = (MODULE_ROOT / "data_engineering" / "build_model_datasets.py").read_text(encoding="utf-8")
        bridge = (MODULE_ROOT / "training" / "chunked_tslib_dataset.py").read_text(encoding="utf-8")
        runner = (MODULE_ROOT / "training" / "run_chunked_tslib_training.py").read_text(encoding="utf-8")
        self.assertIn("WISBURN_DATA_ENGINEERING_ROOT", bridge)
        self.assertIn("WISBURN_FORECAST_ARTIFACT_ROOT", bridge)
        self.assertIn("WISBURN_TSLIB_ROOT", runner)
        self.assertIn("WISBURN_FORECAST_POINT_ROOT", data_builder)
        self.assertIn("WISBURN_FORECAST_CLASSIFICATION_XLSX", data_builder)
        self.assertIn('"offline" / "forecast" / "data_engineering"', bridge)
        self.assertIn('"offline" / "forecast" / "tslib"', runner)
        self.assertNotIn("PROJECT_DIR", bridge)
        self.assertNotIn("PROJECT_DIR", runner)

    def test_report_scripts_read_artifact_root(self) -> None:
        for script in (
            "generate_model_forecast_review.py",
            "generate_model1_10s_forecast_charts.py",
            "generate_smoothed_forecast_review.py",
        ):
            text = (MODULE_ROOT / "reports" / script).read_text(encoding="utf-8")
            self.assertIn('"offline" / "artifacts" / "forecast" / "training_ready"', text)
            self.assertNotIn('"backend" / "pipeline" / "training_ready"', text)


if __name__ == "__main__":
    unittest.main()
