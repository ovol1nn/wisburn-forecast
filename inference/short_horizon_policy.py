"""History-only preheater correction. Inputs and outputs are in degrees Celsius."""

import numpy as np


def apply_policy(history, prediction, config):
    history = np.asarray(history, dtype='float64')
    prediction = np.asarray(prediction, dtype='float64')
    if history.ndim != 2 or history.shape[1] != 120 or prediction.shape != (len(history), 60):
        raise ValueError('Expected history [batch,120] and prediction [batch,60]')
    result = prediction.copy()
    gate = np.zeros(len(history))
    if not config.get('enabled', False):
        return result, gate
    threshold, end = float(config['threshold_c']), int(config['end_seconds'])
    if not np.isfinite(threshold) or threshold <= 0 or end not in [120, 180]:
        raise ValueError('Invalid short-horizon policy configuration')
    usable = np.isfinite(history[:, -61:]).all(axis=1)
    observed = history[usable]
    # A missing signal must never be interpreted as stable after mean filling.
    score = np.maximum(np.ptp(observed[:, -30:], axis=1), np.abs(observed[:, -1] - observed[:, -61]))
    gate[usable] = np.clip(1. - score / threshold, 0., 1.)
    seconds = np.arange(1, 61) * 10.
    taper = .5 * (1. + np.cos(np.pi * np.minimum(seconds / end, 1.)))
    # Write only active horizons, preserving every later value exactly.
    active = np.flatnonzero(seconds < end)
    keep = np.flatnonzero(gate > 0)
    current = history[keep, -1, None]
    result[np.ix_(keep, active)] = current + (1. - gate[keep, None] * taper[active]) * (prediction[np.ix_(keep, active)] - current)
    return result, gate
