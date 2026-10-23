"""One bounded, three-parameter residual calibration after simple offsets failed."""

import json
import sys

import numpy as np
import pandas as pd

import improve_oxygen_history_only as experiment
from oxygen_correction import correct_oxygen

OUT = experiment.OUT / 'state_calibration'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'frozen_policy.json').exists():
        raise FileExistsError('Do not tune again after replication; use the frozen record')
    a = np.load(experiment.OUT / 'oxygen_evaluation_cache.npz')
    p, y, history, scales = [a[k] for k in ['val_pred', 'val_true', 'val_history', 'target_scale']]
    chunks = json.loads((experiment.OUT / 'source_provenance.json').read_text(encoding='utf-8'))['validation_chunks']
    cal, check = np.isin(a['val_chunks'], chunks[:26]), np.isin(a['val_chunks'], chunks[27:])
    meta = json.loads((experiment.ART / 'metadata.json').read_text(encoding='utf-8'))
    scaler = json.loads((experiment.ART / 'scaler.json').read_text(encoding='utf-8'))
    frozen, trials = {}, []
    for j, code in enumerate(experiment.CODES):
        cp, cy, ch = p[cal, :, j], y[cal, :, j], history[cal, :, j]
        current, scale = ch[:, -1], scales[j]
        mean = scaler['mean'][meta['feature_columns'].index(code)]
        ramp = 1. - np.exp(-np.arange(1, 61) * 10. / 180.)
        design = np.stack([np.broadcast_to(ramp, cp.shape), ramp * (current[:, None] - mean) / scale,
                           (cp - current[:, None]) / scale], axis=-1).reshape(-1, 3)
        residual = ((cy - cp) / scale).ravel()
        base = experiment.metrics(cp, cy, current, .1 * scale)
        eligible = []
        for regularization in [.01, .1, 1.]:
            beta = np.linalg.solve(design.T @ design / len(design) + regularization * np.eye(3), design.T @ residual / len(design))
            config = dict(method='state_linear', training_mean=mean, training_scale=scale,
                          coefficients=beta.tolist(), max_offset=.5 * scale)
            prediction, _ = correct_oxygen(ch, cp, config)
            m = experiment.metrics(prediction, cy, current, .1 * scale)
            passed = m['mae'] <= base['mae'] * .99 and m['rmse'] <= base['rmse'] * 1.02 and m['direction_accuracy'] >= base['direction_accuracy'] - .01
            for mask in [cy[:, -1] - current >= .1 * scale, cy[:, -1] - current <= -.1 * scale,
                         (cy[:, :6] - current[:, None]).max(1) >= .1 * scale,
                         (cy[:, :6] - current[:, None]).min(1) <= -.1 * scale]:
                if mask.any():
                    passed &= np.abs(prediction[mask] - cy[mask]).mean() <= np.abs(cp[mask] - cy[mask]).mean() * 1.02
            trials.append(dict(target=code, regularization=regularization, eligible=bool(passed), config=json.dumps(config), **m))
            if passed:
                eligible.append((m['mae'], config))
        frozen[code] = min(eligible, key=lambda r: r[0])[1] if eligible else dict(method='none')
    pd.DataFrame(trials).to_csv(OUT / 'calibration_trials.csv', index=False, encoding='utf-8-sig')
    (OUT / 'frozen_policy.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Frozen state calibration', frozen, flush=True)
    rows, days, horizons, contexts = [], [], [], []
    for name, pp, yy, hh, ids in [
        ('validation_calibration', p[cal], y[cal], history[cal], a['val_chunks'][cal]),
        ('validation_replication', p[check], y[check], history[check], a['val_chunks'][check]),
        ('test_all', a['test_pred'], a['test_true'], a['test_history'], a['test_chunks'])]:
        r, d, h, c = experiment.evaluate_period(name, pp, yy, hh, ids, frozen, scales)
        rows += r; days += d; horizons += h; contexts += c
    for filename, data in [('group_metrics', rows), ('daily_metrics', days), ('horizon_metrics', horizons), ('historical_state_metrics', contexts)]:
        pd.DataFrame(data).to_csv(OUT / (filename + '.csv'), index=False, encoding='utf-8-sig')
    checks, policy = [], {}
    records = pd.DataFrame(rows)
    for code in experiment.CODES:
        passed_target = frozen[code]['method'] != 'none'
        for period in ['validation_replication', 'test_all']:
            g = records[(records.target == code) & (records.period == period)]
            overall = g[g.group == 'all'].set_index('method')
            raw, corrected = overall.loc['raw'], overall.loc['corrected']
            day = pd.DataFrame(days).query('target == @code and period == @period').pivot(index='chunk_id', columns='method', values='mae')
            passed = corrected.mae < raw.mae and corrected.rmse <= raw.rmse * 1.02 and corrected.direction_accuracy >= raw.direction_accuracy - .01
            event_changes = {}
            for group in ['rising', 'falling', 'rapid_rise_first_minute', 'rapid_fall_first_minute']:
                e = g[g.group == group].set_index('method')
                ratio = e.loc['corrected', 'mae'] / e.loc['raw', 'mae']
                event_changes[group] = float(ratio - 1)
                passed &= ratio <= 1.02
            passed &= (day.corrected <= day.raw + 1e-12).sum() >= len(day) / 2
            passed_target &= bool(passed)
            checks.append(dict(target=code, period=period, passed=bool(passed), improvement=float(1 - corrected.mae / raw.mae),
                               direction_accuracy_change=float(corrected.direction_accuracy - raw.direction_accuracy),
                               event_mae_relative_change=event_changes, nonworse_days=int((day.corrected <= day.raw + 1e-12).sum()),
                               days=len(day), worst_day_mae_relative_change=float((day.corrected / day.raw - 1).max())))
        policy[code] = frozen[code] if passed_target else dict(method='none')
    result = dict(oxygen_policy=policy, checks=checks, new_unseen_test=False, original_weights_unchanged=True,
                  calibration_parameters_per_target=3, no_further_tuning=True)
    (OUT / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
