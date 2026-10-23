"""Diagnose and calibrate two oxygen outputs without retraining or future inputs."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'offline/forecast/inference'))
from predict import HistoryOnlyPredictor
from oxygen_correction import correct_oxygen

ART = ROOT / 'offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60'
DATA = ROOT / 'offline/artifacts/forecast/data_engineering/outputs/datasets/chunked/model1_10s_seq120_pred60'
V1 = ROOT / 'offline/artifacts/forecast/candidates/model1_history_only_guarded_v1_20261022'
OUT = ROOT / 'offline/artifacts/forecast/candidates/model1_oxygen_corrected_v2_20261022'
CODES = ['B3QT00', 'B3_O2_S1']


def source_signature():
    return {n: dict(size=(ART / n).stat().st_size, mtime_ns=(ART / n).stat().st_mtime_ns)
            for n in ['pred.npy', 'true.npy', 'checkpoint.pth', 'scaler.json', 'metadata.json']}


def prepare_data():
    cache = OUT / 'oxygen_evaluation_cache.npz'
    if cache.exists():
        old = json.loads((OUT / 'source_provenance.json').read_text(encoding='utf-8'))
        if old['artifacts'] != source_signature():
            raise ValueError('Source artifacts changed; do not reuse cache')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    signature = source_signature()
    import torch
    torch.set_num_threads(4)
    predictor = HistoryOnlyPredictor(V1 / 'bundle')
    meta = predictor.meta
    ti = [predictor.features.index(c) for c in CODES]
    means, scales = predictor.mean[ti], predictor.scale[ti]
    selected = pd.read_csv(ART / 'evaluation_history_only_20261021/independent_temperature_validation_20261022/selected_windows.csv')
    val_chunks = pd.read_csv(DATA / 'splits/val_chunks.csv')
    test_chunks = pd.read_csv(DATA / 'splits/test_chunks.csv')
    saved_p, saved_y = [np.load(ART / n, mmap_mode='r') for n in ['pred.npy', 'true.npy']]

    def load(row):
        f = pd.read_parquet(DATA / 'frames' / (row.chunk_id + '.parquet'), columns=predictor.features)
        return f.to_numpy(dtype='float32')

    def infer(values, starts):
        z = np.nan_to_num((values - predictor.mean) / predictor.scale, nan=0., posinf=0., neginf=0.).astype('float32')
        result = []
        with torch.inference_mode():
            for offset in range(0, len(starts), 4):
                positions = starts[offset:offset + 4]
                x = torch.from_numpy(np.stack([z[s:s + 120] for s in positions]))
                future = x[:, -1:].repeat(1, 60, 1)
                future[:, :, predictor.inds] = x[:, -60:, predictor.inds].mean(1, keepdim=True)
                decoder = torch.cat([x[:, -60:], future], dim=1)
                mask = torch.zeros(len(x), 60, len(predictor.features), dtype=torch.bool)
                mask[:, :, predictor.inds] = True
                out = predictor.model(x, None, decoder, None, dec_mask=mask).numpy()[:, :, ti]
                result.append(out.astype('float64') * scales + means)
        return np.concatenate(result)

    parity, offset = [], 0
    for i, row in enumerate(test_chunks.itertuples()):
        if i in [0, len(test_chunks) // 2, len(test_chunks) - 1]:
            values = load(row)
            w = pd.read_csv(DATA / 'windows' / (row.chunk_id + '_windows.csv'))
            ix = np.unique([0, len(w) // 2, len(w) - 1])
            p = infer(values, w.input_start_pos.to_numpy()[ix])
            reference = np.asarray(saved_p[offset + ix])[:, :, ti].astype('float64') * scales + means
            np.testing.assert_allclose(p, reference, rtol=1e-4, atol=1e-4)
            parity.append(dict(chunk_id=row.chunk_id, windows=len(ix), max_error_pp=float(np.abs(p - reference).max())))
        offset += int(row.windows)
    print('Oxygen real-weight prediction parity passed', flush=True)
    vp, vy, vh = np.empty((len(selected), 60, 2)), np.empty((len(selected), 60, 2)), np.empty((len(selected), 120, 2))
    for i, row in enumerate(val_chunks.itertuples()):
        keep = (selected.chunk_id == row.chunk_id).to_numpy()
        values = load(row)
        starts = selected.loc[keep, 'input_start_pos'].to_numpy()
        vp[keep] = infer(values, starts)
        vh[keep] = np.stack([values[s:s + 120, ti] for s in starts])
        vy[keep] = np.stack([values[s + 120:s + 180, ti] for s in starts])
        print(f'Validation oxygen inference {i + 1}/{len(val_chunks)}', flush=True)
    n = len(saved_p)
    tp, ty, th = np.empty((n, 60, 2)), np.empty((n, 60, 2)), np.empty((n, 120, 2))
    test_ids, offset = [], 0
    for i, row in enumerate(test_chunks.itertuples()):
        values = pd.read_parquet(DATA / 'frames' / (row.chunk_id + '.parquet'), columns=CODES).to_numpy()
        w = pd.read_csv(DATA / 'windows' / (row.chunk_id + '_windows.csv'))
        starts = w.input_start_pos.to_numpy()
        stop = offset + len(w)
        th[offset:stop] = np.stack([values[s:s + 120] for s in starts])
        tp[offset:stop] = np.asarray(saved_p[offset:stop])[:, :, ti].astype('float64') * scales + means
        ty[offset:stop] = np.asarray(saved_y[offset:stop])[:, :, ti].astype('float64') * scales + means
        test_ids.extend([row.chunk_id] * len(w))
        offset = stop
        print(f'Test oxygen extraction {i + 1}/{len(test_chunks)}', flush=True)
    assert offset == n and all(np.isfinite(z).all() for z in [vp, vy, vh, tp, ty, th])
    np.savez_compressed(cache, val_pred=vp, val_true=vy, val_history=vh, test_pred=tp, test_true=ty,
                        test_history=th, val_chunks=selected.chunk_id.to_numpy(dtype=str),
                        test_chunks=np.array(test_ids), target_scale=scales)
    assert signature == source_signature()
    provenance = dict(artifacts=signature, inference_parity=parity,
                      checkpoint_sha256=hashlib.sha256((ART / 'checkpoint.pth').read_bytes()).hexdigest(),
                      original_artifacts_unchanged=True, validation_chunks=val_chunks.chunk_id.tolist())
    (OUT / 'source_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(p, y, current, threshold):
    error = p - y
    sst = np.square(y - y.mean()).sum()
    delta = y[:, -1] - current
    predicted_delta = p[:, -1] - current
    eligible = np.abs(delta) >= threshold
    denominator = np.square(delta[eligible]).sum()
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                window_mae_p95=float(np.quantile(np.abs(error).mean(axis=1), .95)),
                bias=float(error.mean()), r2=float(1 - np.square(error).sum() / sst) if sst else None,
                direction_eligible=int(eligible.sum()),
                direction_accuracy=float((np.sign(delta[eligible]) == np.sign(predicted_delta[eligible])).mean()) if eligible.any() else None,
                endpoint_change_gain=float(np.sum(delta[eligible] * predicted_delta[eligible]) / denominator) if denominator else None)


def evaluate_period(name, p, y, h, chunk_ids, configs, scales):
    records, daily, horizons, scenarios = [], [], [], []
    for j, code in enumerate(CODES):
        current, threshold = h[:, -1, j], .1 * scales[j]
        corrected, _ = correct_oxygen(h[:, :, j], p[:, :, j], configs[code])
        true_delta = y[:, -1, j] - current
        groups = {'all': np.ones(len(y), dtype=bool), 'rising': true_delta >= threshold,
                  'falling': true_delta <= -threshold, 'small_change': np.abs(true_delta) < threshold,
                  'rapid_rise_first_minute': (y[:, :6, j] - current[:, None]).max(axis=1) >= threshold,
                  'rapid_fall_first_minute': (y[:, :6, j] - current[:, None]).min(axis=1) <= -threshold}
        for group, keep in groups.items():
            if not keep.any():
                continue
            for method, prediction in [('raw', p[:, :, j]), ('corrected', corrected), ('persistence', np.broadcast_to(current[:, None], corrected.shape))]:
                records.append(dict(period=name, target=code, group=group, method=method, windows=int(keep.sum()),
                                    **metrics(prediction[keep], y[keep, :, j], current[keep], threshold)))
        for chunk in np.unique(chunk_ids):
            keep = chunk_ids == chunk
            for method, prediction in [('raw', p[:, :, j]), ('corrected', corrected)]:
                daily.append(dict(period=name, target=code, chunk_id=chunk, method=method,
                                  **metrics(prediction[keep], y[keep, :, j], current[keep], threshold)))
        for method, prediction in [('raw', p[:, :, j]), ('corrected', corrected), ('persistence', np.broadcast_to(current[:, None], corrected.shape))]:
            for step in range(60):
                e = prediction[:, step] - y[:, step, j]
                horizons.append(dict(period=name, target=code, method=method, seconds=(step + 1) * 10,
                                     mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.square(e).mean())), bias=float(e.mean())))
        # Observable history groups, fixed thresholds based on training standard deviation.
        historical_delta = h[:, -1, j] - h[:, -31, j]
        for label, keep in [('history_rising', historical_delta >= threshold), ('history_falling', historical_delta <= -threshold), ('history_stable', np.abs(historical_delta) < threshold)]:
            for method, prediction in [('raw', p[:, :, j]), ('corrected', corrected)]:
                if keep.any():
                    scenarios.append(dict(period=name, target=code, group=label, method=method, windows=int(keep.sum()), **metrics(prediction[keep], y[keep, :, j], current[keep], threshold)))
    return records, daily, horizons, scenarios


def calibrate_and_check():
    if (OUT / 'frozen_oxygen_policy.json').exists():
        raise FileExistsError('Preserve frozen policy; do not retune after replication')
    a = np.load(OUT / 'oxygen_evaluation_cache.npz')
    p, y, h, scales = [a[n] for n in ['val_pred', 'val_true', 'val_history', 'target_scale']]
    chunks = json.loads((OUT / 'source_provenance.json').read_text(encoding='utf-8'))['validation_chunks']
    calibration = np.isin(a['val_chunks'], chunks[:26])
    replication = np.isin(a['val_chunks'], chunks[27:])
    trials, selected = [], {}
    for j, code in enumerate(CODES):
        cp, cy, ch = p[calibration, :, j], y[calibration, :, j], h[calibration, :, j]
        current, threshold = ch[:, -1], .1 * scales[j]
        limit = max(float(np.quantile(np.abs(current - cp[:, 0]), .99)), .01)
        candidates = [dict(method='none'), dict(method='constant_bias', bias=float(np.median(cy - cp)), max_offset=limit)]
        candidates += [dict(method='origin_anchor', strength=s, decay_seconds=t, max_offset=limit) for s in [.5, 1.] for t in [180, 600]]
        baseline = metrics(cp, cy, current, threshold)
        eligible = []
        for config in candidates:
            pred, _ = correct_oxygen(ch, cp, config)
            m = metrics(pred, cy, current, threshold)
            passed = m['mae'] <= baseline['mae'] * .99 and m['rmse'] <= baseline['rmse'] * 1.02 and m['direction_accuracy'] >= baseline['direction_accuracy'] - .01
            for keep in [cy[:, -1] - current >= threshold, cy[:, -1] - current <= -threshold,
                         (cy[:, :6] - current[:, None]).max(axis=1) >= threshold,
                         (cy[:, :6] - current[:, None]).min(axis=1) <= -threshold]:
                if keep.any():
                    passed &= np.abs(pred[keep] - cy[keep]).mean() <= np.abs(cp[keep] - cy[keep]).mean() * 1.02
            trials.append(dict(target=code, config=json.dumps(config), eligible=bool(passed), **m))
            if passed:
                eligible.append((m['mae'], config))
        selected[code] = min(eligible, key=lambda item: item[0])[1] if eligible else dict(method='none')
    pd.DataFrame(trials).to_csv(OUT / 'calibration_trials.csv', index=False, encoding='utf-8-sig')
    (OUT / 'frozen_oxygen_policy.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Frozen oxygen policies:', selected, flush=True)
    rows, days, horizon, scenarios = [], [], [], []
    for name, mask in [('validation_calibration', calibration), ('validation_replication', replication)]:
        r, d, hh, s = evaluate_period(name, p[mask], y[mask], h[mask], a['val_chunks'][mask], selected, scales)
        rows += r; days += d; horizon += hh; scenarios += s
    tp, ty, th = [a[n] for n in ['test_pred', 'test_true', 'test_history']]
    r, d, hh, s = evaluate_period('test_all', tp, ty, th, a['test_chunks'], selected, scales)
    rows += r; days += d; horizon += hh; scenarios += s
    for filename, records in [('group_metrics', rows), ('daily_metrics', days), ('horizon_metrics', horizon), ('historical_state_metrics', scenarios)]:
        pd.DataFrame(records).to_csv(OUT / (filename + '.csv'), index=False, encoding='utf-8-sig')
    checks, packaged = [], {}
    frame = pd.DataFrame(rows)
    for code in CODES:
        passes = selected[code]['method'] != 'none'
        for period in ['validation_replication', 'test_all']:
            g = frame[(frame.target == code) & (frame.period == period)]
            all_group = g[g.group == 'all'].set_index('method')
            raw, corrected = all_group.loc['raw'], all_group.loc['corrected']
            day = pd.DataFrame(days).query('target == @code and period == @period').pivot(index='chunk_id', columns='method', values='mae')
            passed = corrected.mae < raw.mae and corrected.rmse <= raw.rmse * 1.02 and corrected.direction_accuracy >= raw.direction_accuracy - .01
            for group in ['rising', 'falling', 'rapid_rise_first_minute', 'rapid_fall_first_minute']:
                event = g[g.group == group].set_index('method')
                passed &= event.loc['corrected', 'mae'] <= event.loc['raw', 'mae'] * 1.02
            passed &= (day.corrected <= day.raw + 1e-12).sum() >= len(day) / 2
            passes &= bool(passed)
            checks.append(dict(target=code, period=period, passed=bool(passed), improvement=float(1 - corrected.mae / raw.mae),
                               nonworse_days=int((day.corrected <= day.raw + 1e-12).sum()), days=len(day)))
        packaged[code] = selected[code] if passes else dict(method='none')
    result = dict(oxygen_policy=packaged, checks=checks, new_unseen_test=False, original_weights_unchanged=True)
    (OUT / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if '--evaluate-only' in sys.argv:
        calibrate_and_check()
    else:
        prepare_data()
        calibrate_and_check()
