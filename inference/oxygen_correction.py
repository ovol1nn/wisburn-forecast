"""Experimental oxygen calibrations; failed acceptance and NOT enabled in v1.

Units are percentage points. These pure functions do not receive future truth.
"""

import numpy as np


def correct_oxygen(history, prediction, config):
    history = np.asarray(history, dtype='float64')
    prediction = np.asarray(prediction, dtype='float64')
    if history.ndim != 2 or history.shape[1] != 120 or prediction.shape != (len(history), 60):
        raise ValueError('Expected history [batch,120] and prediction [batch,60]')
    result, offset = prediction.copy(), np.zeros(len(history))
    method = config.get('method', 'none')
    if method == 'none':
        return result, offset
    limit = float(config['max_offset'])
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('Invalid oxygen offset limit')
    usable = np.isfinite(history[:, -6:]).all(axis=1) & np.isfinite(prediction).all(axis=1)
    if method == 'state_linear':
        mean, scale = float(config['training_mean']), float(config['training_scale'])
        coefficients = np.asarray(config['coefficients'], dtype='float64')
        if not np.isfinite([mean, scale]).all() or scale <= 0 or coefficients.shape != (3,) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid oxygen state calibration')
        current = history[usable, -1, None]
        ramp = 1. - np.exp(-np.arange(1, 61) * 10. / 180.)
        correction = (coefficients[0] * ramp + coefficients[1] * ramp * (current - mean) / scale
                      + coefficients[2] * (prediction[usable] - current) / scale) * scale
        correction = np.clip(correction, -limit, limit)
        result[usable] += correction
        offset[usable] = correction[:, 0]
        return result, offset
    if method == 'constant_bias':
        bias = float(config['bias'])
        if not np.isfinite(bias):
            raise ValueError('Invalid oxygen bias')
        offset[usable] = np.clip(bias, -limit, limit)
        decay = np.ones(60)
    elif method == 'origin_anchor':
        strength, seconds = float(config['strength']), float(config['decay_seconds'])
        if strength not in [.5, 1.] or seconds not in [180., 600.]:
            raise ValueError('Invalid oxygen anchor configuration')
        offset[usable] = strength * np.clip(history[usable, -1] - prediction[usable, 0], -limit, limit)
        decay = np.exp(-(np.arange(60) * 10.) / seconds)
    else:
        raise ValueError('Unknown oxygen correction method')
    result[usable] += offset[usable, None] * decay
    return result, offset
