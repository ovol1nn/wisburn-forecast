from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from oxygen_correction import correct_oxygen


class OxygenCorrectionTests(unittest.TestCase):
    def test_anchor_corrects_offset_without_flattening_trend(self):
        history = np.full((1, 120), 3.)
        prediction = np.linspace(2.8, 3.4, 60)[None]
        config = dict(method='origin_anchor', strength=1., decay_seconds=600, max_offset=.5)
        result, offset = correct_oxygen(history, prediction, config)
        self.assertAlmostEqual(offset[0], .2)
        self.assertGreater(result[0, -1], result[0, 0])
        self.assertLess(result[0, -1] - prediction[0, -1], .2)
        np.testing.assert_array_equal(prediction, np.linspace(2.8, 3.4, 60)[None])

    def test_constant_bias_preserves_all_step_differences(self):
        h, p = np.ones((2, 120)), np.arange(120).reshape(2, 60).astype(float)
        r, _ = correct_oxygen(h, p, dict(method='constant_bias', bias=.1, max_offset=.5))
        np.testing.assert_allclose(np.diff(r), np.diff(p), atol=1e-13)

    def test_missing_recent_history_or_nonfinite_prediction_disables(self):
        h, p = np.ones((2, 120)), np.ones((2, 60)) * 2
        h[0, -2] = np.nan
        p[1, 10] = np.nan
        r, offset = correct_oxygen(h, p, dict(method='origin_anchor', strength=1., decay_seconds=600, max_offset=.5))
        np.testing.assert_array_equal(r, p)
        np.testing.assert_array_equal(offset, [0., 0.])

    def test_offset_capped_and_identity_available(self):
        h, p = np.ones((1, 120)), np.ones((1, 60)) * 5
        _, delta = correct_oxygen(h, p, dict(method='origin_anchor', strength=1., decay_seconds=180, max_offset=.25))
        self.assertAlmostEqual(delta[0], -.25)
        r, _ = correct_oxygen(h, p, dict(method='none'))
        np.testing.assert_array_equal(r, p)

    def test_bad_configuration_rejected(self):
        with self.assertRaises(ValueError):
            correct_oxygen(np.ones((1, 120)), np.ones((1, 60)), dict(method='origin_anchor', strength=3., decay_seconds=0, max_offset=-1))

    def test_state_calibration_identity_and_bound(self):
        h, p = np.ones((1, 120)) * 3, np.linspace(3., 5., 60)[None]
        config = dict(method='state_linear', training_mean=3., training_scale=1., coefficients=[0., 0., 0.], max_offset=.2)
        result, _ = correct_oxygen(h, p, config)
        np.testing.assert_array_equal(result, p)
        config['coefficients'] = [10., 0., 1.]
        result, _ = correct_oxygen(h, p, config)
        self.assertLessEqual(np.abs(result - p).max(), .2 + 1e-12)


if __name__ == '__main__':
    unittest.main()
