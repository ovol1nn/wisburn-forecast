from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch


TSLIB_ROOT = Path(__file__).resolve().parents[1] / "tslib"
sys.path.insert(0, str(TSLIB_ROOT))

# The tested decoder helper does not use DTW. Keep the focused test runnable
# when the optional SciPy evaluation dependency is absent.
if importlib.util.find_spec("scipy") is None:
    scipy = ModuleType("scipy")
    spatial = ModuleType("scipy.spatial")
    distance = ModuleType("scipy.spatial.distance")
    distance.cdist = lambda *args, **kwargs: None
    scipy.spatial = spatial
    spatial.distance = distance
    sys.modules.update({"scipy": scipy, "scipy.spatial": spatial,
                        "scipy.spatial.distance": distance})

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402


class HistoryOnlyDecodingTests(unittest.TestCase):
    def test_future_observations_do_not_change_decoder_input(self) -> None:
        experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        experiment.args = SimpleNamespace(
            label_len=2,
            pred_len=3,
            no_future_covariates=True,
            loss_target_indices=[1],
        )
        batch_x = torch.tensor([[[1.0, 10.0], [2.0, 14.0], [7.0, 20.0]]])
        batch_y = torch.tensor([[[3.0, 30.0], [4.0, 50.0], [5.0, 60.0],
                                 [6.0, 70.0], [8.0, 90.0]]])
        changed_future = batch_y.clone()
        changed_future[:, -3:, :] = 999999.0

        decoder, mask = experiment._build_crossformer_decoder_input_and_mask(batch_y, batch_x)
        changed_decoder, changed_mask = experiment._build_crossformer_decoder_input_and_mask(
            changed_future, batch_x)

        self.assertTrue(torch.equal(decoder, changed_decoder))
        self.assertTrue(torch.equal(mask, changed_mask))
        self.assertEqual(decoder[0, 2:, 0].tolist(), [7.0, 7.0, 7.0])
        self.assertEqual(decoder[0, 2:, 1].tolist(), [40.0, 40.0, 40.0])
        self.assertTrue(mask[0, :, 1].all())


if __name__ == "__main__":
    unittest.main()
