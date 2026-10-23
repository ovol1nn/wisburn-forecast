"""Read-only diagnostics of saved Model 1 forecasts; no model fitting or deployment."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60'
DATA = ROOT / 'offline/artifacts/forecast/data_engineering/outputs/datasets/chunked/model1_10s_seq120_pred60'
OUT = ART / 'evaluation_history_only_20261021/diagnostics_20261021'
CACHE = ROOT / 'offline/artifacts/forecast/data_engineering/outputs/cache/resampled_10s'
NAMES = ['主蒸汽流量', '氧量A', '氧量B', '中上部炉膛温度', '垃圾层厚度', '汽包水位1', '汽包水位2', '汽包水位3', '炉膛负压A', '炉膛负压B', '主蒸汽压力A', '主蒸汽压力B', '一次风预热器出口温度', '主蒸汽温度']
UNITS = ['t/h', '%', '%', '°C', '%', 'mm', 'mm', 'mm', 'Pa', 'Pa', 'MPa', 'MPa', '°C', '°C']
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def save_fig(name, fig):
    fig.savefig(OUT / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def load_chunk(row, columns):
    frame = pd.read_parquet(DATA / 'frames' / Path(row['frame_path']).name, columns=columns)
    windows = pd.read_csv(DATA / 'windows' / Path(row['window_path']).name)
    if len(windows) != int(row['windows']):
        raise ValueError('window count mismatch: ' + row['chunk_id'])
    return frame, windows


def summarize(p, y, origin, selected, targets, label):
    rows = []
    for j, target in enumerate(targets):
        e = p[selected, :, j] - y[selected, :, j]
        base = origin[selected, j, None] - y[selected, :, j]
        actual = y[selected, :, j]
        sst = np.square(actual - actual.mean()).sum()
        rows.append({'group': label, 'target': target, 'name': NAMES[j], 'windows': int(selected.sum()), 'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt(np.square(e).mean())), 'bias': float(e.mean()), 'r2': float(1 - np.square(e).sum() / sst) if sst else np.nan, 'baseline_mae': float(np.abs(base).mean()), 'mae_skill': float(1 - np.abs(e).sum() / np.abs(base).sum())})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_state = {name: {'size': (ART / name).stat().st_size, 'mtime_ns': (ART / name).stat().st_mtime_ns} for name in ['pred.npy', 'true.npy', 'metadata.json', 'scaler.json', 'checkpoint.pth']}
    meta = json.loads((ART / 'metadata.json').read_text(encoding='utf-8'))
    sc = json.loads((ART / 'scaler.json').read_text(encoding='utf-8'))
    features, targets = meta['feature_columns'], meta['target_columns']
    inds = np.array(meta['target_indices'])
    mean, scale = np.asarray(sc['mean']), np.asarray(sc['scale'])
    tm, ts = mean[inds], scale[inds]
    original_p = np.load(ART / 'pred.npy', mmap_mode='r')
    original_y = np.load(ART / 'true.npy', mmap_mode='r')
    if original_p.shape != original_y.shape or original_p.shape[1:] != (60, len(features)):
        raise ValueError('saved array shapes do not match metadata')
    chunks = csv_rows(DATA / 'splits/test_chunks.csv')
    train_chunks = csv_rows(DATA / 'splits/train_chunks.csv')
    n, h, k = original_p.shape[0], 60, len(targets)
    if sum(int(c['windows']) for c in chunks) != n:
        raise ValueError('test window count mismatch')
    # Target-only caches use about 1.4 GB on disk in total, keeping memory bounded.
    p = np.lib.format.open_memmap(OUT / '_target_pred.npy', mode='w+', dtype='float32', shape=(n, h, k))
    y = np.lib.format.open_memmap(OUT / '_target_true.npy', mode='w+', dtype='float32', shape=(n, h, k))
    origin = np.empty((n, k), dtype='float64')
    hist_delta = np.empty_like(origin)
    past_actions = np.empty(n)
    future_actions = np.empty(n)
    input_missing_fraction = np.empty(n)
    current_load = np.empty(n)
    chunk_number = np.empty(n, dtype='int16')
    target_times = np.empty(n, dtype='datetime64[ns]')
    daily_rows, coverage_rows, input_quality_rows = [], [], []
    per_feature_missing = np.zeros(len(features), dtype='int64')
    used_history_rows = 0
    point_map = pd.read_csv(ROOT / 'assets/profiles/xingrong_3/points/point_map.csv').fillna('')
    controls = point_map.loc[(point_map['role'] == 'control') & point_map['point_code'].isin(features), 'point_code'].drop_duplicates().tolist()
    ci = [features.index(c) for c in controls]
    training_load = []
    for c in train_chunks:
        f = pd.read_parquet(DATA / 'frames' / Path(c['frame_path']).name, columns=['timestamp', targets[0], 'valid_for_training'])
        selected = (f['timestamp'] >= pd.Timestamp(c['core_start'])) & (f['timestamp'] < pd.Timestamp(c['core_end'])) & f['valid_for_training'].fillna(False)
        training_load.append(f.loc[selected, targets[0]].dropna().to_numpy())
    load_cuts = np.quantile(np.concatenate(training_load), [1 / 3, 2 / 3])
    del training_load
    cursor = 0
    worst_cases = {j: (-np.inf, None) for j in [0, 8, 12]}
    for day, c in enumerate(chunks):
        frame, w = load_chunk(c, ['timestamp', *features, 'valid_for_training'])
        x = frame[features].to_numpy(dtype='float64')
        pos = w['input_end_pos'].to_numpy(dtype=int)
        starts = w['input_start_pos'].to_numpy(dtype=int)
        ends = w['target_end_pos'].to_numpy(dtype=int)
        count = len(w)
        rows = slice(cursor, cursor + count)
        chunk_number[rows] = day
        target_times[rows] = pd.to_datetime(w['target_end_time']).to_numpy(dtype='datetime64[ns]')
        origin[rows] = x[pos][:, inds]
        hist_delta[rows] = x[pos][:, inds] - x[np.maximum(pos - 30, starts)][:, inds]
        current_load[rows] = x[pos, inds[0]]
        missing = ~np.isfinite(x)
        per_row_missing = missing.sum(axis=1)
        cum_missing = np.r_[0, per_row_missing.cumsum()]
        input_missing_fraction[rows] = (cum_missing[pos + 1] - cum_missing[starts]) / (120 * len(features))
        history_cover = np.zeros(len(frame) + 1, dtype=int)
        np.add.at(history_cover, starts, 1)
        np.add.at(history_cover, pos + 1, -1)
        history_cover = history_cover[:-1].cumsum() > 0
        per_feature_missing += missing[history_cover].sum(axis=0)
        used_history_rows += int(history_cover.sum())
        control_values = pd.DataFrame(x[:, ci])
        historical_range = (control_values.rolling(31, min_periods=31).max() - control_values.rolling(31, min_periods=31).min()).to_numpy() / scale[ci]
        future_range = (control_values.rolling(61, min_periods=61).max() - control_values.rolling(61, min_periods=61).min()).to_numpy() / scale[ci]
        past_actions[rows] = np.nanmean(historical_range[pos], axis=1)
        future_actions[rows] = np.nanmean(future_range[ends], axis=1)
        for local in range(0, count, 256):
            end = min(local + 256, count)
            block = slice(cursor + local, cursor + end)
            p[block] = np.asarray(original_p[block][:, :, inds], dtype='float64') * ts + tm
            y[block] = np.asarray(original_y[block][:, :, inds], dtype='float64') * ts + tm
        if not np.isfinite(p[rows]).all() or not np.isfinite(y[rows]).all() or not np.isfinite(origin[rows]).all():
            raise ValueError('nonfinite targets in ' + c['chunk_id'])
        # First/middle/last windows of every chunk, all 60 future steps and all targets.
        for wi in [0, count // 2, count - 1]:
            truth_frame = x[int(w.iloc[wi]['target_start_pos']):int(w.iloc[wi]['target_end_pos']) + 1, inds]
            if not np.allclose(y[cursor + wi], truth_frame, rtol=2e-5, atol=2e-4):
                raise ValueError('window ordering/alignment mismatch in ' + c['chunk_id'])
        pe = p[rows].astype('float64') - y[rows]
        be = origin[rows, None, :] - y[rows]
        for j, target in enumerate(targets):
            ya = y[rows, :, j].astype('float64')
            sst = np.square(ya - ya.mean()).sum()
            daily_rows.append({'date': c['core_start'][:10], 'chunk': day, 'target': target, 'name': NAMES[j], 'windows': count, 'model_abs_sum': float(np.abs(pe[:, :, j]).sum()), 'baseline_abs_sum': float(np.abs(be[:, :, j]).sum()), 'mae': float(np.abs(pe[:, :, j]).mean()), 'bias': float(pe[:, :, j].mean()), 'r2': float(1 - np.square(pe[:, :, j]).sum() / sst) if sst else np.nan, 'mae_skill': float(1 - np.abs(pe[:, :, j]).sum() / np.abs(be[:, :, j]).sum())})
        for j in worst_cases:
            errors = np.abs(pe[:, :, j]).mean(axis=1)
            loc = int(np.argmax(errors))
            if errors[loc] > worst_cases[j][0]:
                worst_cases[j] = (float(errors[loc]), cursor + loc)
        footprint = np.zeros(len(frame) + 1, dtype=int)
        np.add.at(footprint, w['target_start_pos'].to_numpy(dtype=int), 1)
        np.add.at(footprint, ends + 1, -1)
        unique_future_rows = int((footprint[:-1].cumsum() > 0).sum())
        coverage_rows.append({'date': c['core_start'][:10], 'windows': count, 'origin_hours': count * 10 / 3600, 'unique_future_hours': unique_future_rows * 10 / 3600, 'core_rows': int(c['core_rows']), 'missing_history_windows': int((input_missing_fraction[rows] > 0).sum()), 'mean_input_missing_fraction': float(input_missing_fraction[rows].mean())})
        cursor += count
        print(f'diagnostic arrays: {day + 1}/{len(chunks)} chunks', flush=True)
    p.flush()
    y.flush()
    write_csv('daily_metrics.csv', daily_rows)
    write_csv('test_coverage.csv', coverage_rows)
    for j, code in enumerate(features):
        input_quality_rows.append({'feature': code, 'missing_unique_used_history_rows': int(per_feature_missing[j]), 'used_history_rows': used_history_rows, 'missing_rate': float(per_feature_missing[j] / used_history_rows)})
    write_csv('input_missingness.csv', sorted(input_quality_rows, key=lambda r: -r['missing_rate']))

    # Window statistics and all 60 forecast horizons.
    win_mae = np.empty((n, k), dtype='float32')
    win_base = np.empty_like(win_mae)
    win_bias = np.empty_like(win_mae)
    h_abs, h_base, h_sq, h_basq, h_err = [np.zeros((h, k)) for _ in range(5)]
    diff_pred, diff_true = np.zeros((59, k)), np.zeros((59, k))
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        b = slice(start, stop)
        e = p[b].astype('float64') - y[b]
        d = y[b].astype('float64') - origin[b, None, :]
        win_mae[b] = np.abs(e).mean(axis=1)
        win_base[b] = np.abs(d).mean(axis=1)
        win_bias[b] = e.mean(axis=1)
        h_abs += np.abs(e).sum(axis=0)
        h_base += np.abs(d).sum(axis=0)
        h_sq += np.square(e).sum(axis=0)
        h_basq += np.square(d).sum(axis=0)
        h_err += e.sum(axis=0)
        diff_pred += np.abs(np.diff(p[b].astype('float64'), axis=1)).sum(axis=0)
        diff_true += np.abs(np.diff(y[b].astype('float64'), axis=1)).sum(axis=0)
    hrows = []
    for hi in range(h):
        for j, target in enumerate(targets):
            hrows.append({'target': target, 'name': NAMES[j], 'seconds': (hi + 1) * 10, 'mae': h_abs[hi, j] / n, 'baseline_mae': h_base[hi, j] / n, 'mae_skill': 1 - h_abs[hi, j] / h_base[hi, j], 'mse_skill': 1 - h_sq[hi, j] / h_basq[hi, j], 'bias': h_err[hi, j] / n})
    write_csv('horizon_metrics.csv', hrows)

    # Calibration and later replication both lie INSIDE the already-examined test set.
    # They are exploratory temporal checks, not an untouched final test or original val set.
    cal = chunk_number < 26
    check = chunk_number >= 27
    alpha = np.ones((3, k))
    group_h = [slice(0, 12), slice(12, 30), slice(30, 60)]
    bias_cal = np.empty((3, k))
    for gi, hs in enumerate(group_h):
        dp = p[cal, hs, :].astype('float64') - origin[cal, None, :]
        dy = y[cal, hs, :].astype('float64') - origin[cal, None, :]
        denom = np.square(dp).sum(axis=(0, 1))
        alpha[gi] = np.clip(np.divide((dp * dy).sum(axis=(0, 1)), denom, out=np.zeros(k), where=denom > 0), 0, 1)
        bias_cal[gi] = (dp - dy).mean(axis=(0, 1))
    rule_rows, candidate_rows = [], []
    for gi, label in enumerate(['10–120秒', '130–300秒', '310–600秒']):
        for j, target in enumerate(targets):
            rule_rows.append({'target': target, 'name': NAMES[j], 'horizon_group': label, 'model_weight': alpha[gi, j], 'diagnostic_calibration_bias': bias_cal[gi, j]})
    write_csv('exploratory_rules.csv', rule_rows)
    ep = p[check].astype('float64')
    ey = y[check].astype('float64')
    eo = origin[check]
    candidates = {'原模型': ep, '保持当前值': np.broadcast_to(eo[:, None, :], ep.shape)}
    blend, corrected = ep.copy(), ep.copy()
    for gi, hs in enumerate(group_h):
        blend[:, hs] = eo[:, None, :] + alpha[gi] * (ep[:, hs] - eo[:, None, :])
        corrected[:, hs] -= bias_cal[gi]
    candidates['分时距组合'] = blend
    candidates['分时距偏差校正'] = corrected
    # Fixed 3-point trailing output filter, anchored in the historical last value.
    # Uses only predicted values + current observation, never true future values.
    padded = np.concatenate([eo[:, None, :], eo[:, None, :], ep], axis=1)
    candidates['预测输出3点平滑'] = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3
    all_check = np.ones(len(ep), dtype=bool)
    for name, pred_candidate in candidates.items():
        candidate_rows.extend(summarize(pred_candidate, ey, eo, all_check, targets, name))
    write_csv('exploratory_replication.csv', candidate_rows)
    del ep, ey, eo, candidates, blend, corrected, padded

    # Context groups use only current/history quantities. Future-action and reversal
    # groups are explicitly retrospective and cannot serve as online decision rules.
    cuts_past = np.quantile(past_actions[cal], [1 / 3, 2 / 3])
    cuts_future = np.quantile(future_actions[cal], [1 / 3, 2 / 3])
    group_defs = {'load': np.digitize(current_load, load_cuts), 'past_control_range': np.digitize(past_actions, cuts_past), 'future_control_range_retrospective': np.digitize(future_actions, cuts_future), 'input_missing': (input_missing_fraction > 0).astype(int)}
    grouped_rows = []
    for group_name, labels in group_defs.items():
        for label in np.unique(labels):
            chosen = labels == label
            for j, target in enumerate(targets):
                grouped_rows.append({'group_type': group_name, 'group': int(label), 'target': target, 'name': NAMES[j], 'windows': int(chosen.sum()), 'mae': float(win_mae[chosen, j].mean()), 'baseline_mae': float(win_base[chosen, j].mean()), 'mae_skill': float(1 - win_mae[chosen, j].sum(dtype='float64') / win_base[chosen, j].sum(dtype='float64'))})
    future_delta = y[:, -1, :].astype('float64') - origin
    predicted_delta = p[:, -1, :].astype('float64') - origin
    scenario_rows, target_rows = [], []
    for j, target in enumerate(targets):
        threshold = 0.1 * ts[j]
        hd, fd = hist_delta[:, j], future_delta[:, j]
        history_large, future_large = np.abs(hd) >= threshold, np.abs(fd) >= threshold
        scenarios = np.full(n, '变化较小', dtype=object)
        scenarios[~history_large & future_large] = '新出现变化'
        scenarios[history_large & ~future_large] = '原趋势减弱'
        scenarios[history_large & future_large & (np.sign(hd) == np.sign(fd))] = '趋势延续'
        scenarios[history_large & future_large & (np.sign(hd) != np.sign(fd))] = '趋势反转'
        for label in pd.unique(scenarios):
            chosen = scenarios == label
            scenario_rows.append({'target': target, 'name': NAMES[j], 'scenario': label, 'windows': int(chosen.sum()), 'fraction': float(chosen.mean()), 'mae': float(win_mae[chosen, j].mean()), 'baseline_mae': float(win_base[chosen, j].mean()), 'mae_skill': float(1 - win_mae[chosen, j].sum(dtype='float64') / win_base[chosen, j].sum(dtype='float64'))})
        eligible = future_large
        phase_p = np.array([diff_pred[np.arange(59) % 8 == phase, j].mean() for phase in range(8)])
        phase_t = np.array([diff_true[np.arange(59) % 8 == phase, j].mean() for phase in range(8)])
        boundary = np.arange(59) % 8 == 3
        target_rows.append({'target': target, 'name': NAMES[j], 'unit': UNITS[j], 'mae': float(win_mae[:, j].mean()), 'baseline_mae': float(win_base[:, j].mean()), 'mae_skill': float(1 - win_mae[:, j].sum(dtype='float64') / win_base[:, j].sum(dtype='float64')), 'mae_p95_window': float(np.quantile(win_mae[:, j], .95)), 'worst_5pct_error_share': float(np.sort(win_mae[:, j])[-int(np.ceil(.05 * n)):].sum(dtype='float64') / win_mae[:, j].sum(dtype='float64')), 'window_offset_mse_share': float((60 * np.square(win_bias[:, j]).sum(dtype='float64')) / h_sq[:, j].sum()), 'eligible_direction_windows': int(eligible.sum()), 'direction_accuracy_from_origin': float((np.sign(predicted_delta[eligible, j]) == np.sign(fd[eligible])).mean()), 'endpoint_change_gain': float(np.dot(predicted_delta[:, j], fd) / np.dot(fd, fd)), 'pred_total_variation_ratio': float(diff_pred[:, j].sum() / diff_true[:, j].sum()), 'pred_boundary_jump_ratio': float(diff_pred[boundary, j].mean() / diff_pred[~boundary, j].mean()), 'true_boundary_jump_ratio': float(diff_true[boundary, j].mean() / diff_true[~boundary, j].mean()), 'pred_patch_phases': phase_p.tolist(), 'true_patch_phases': phase_t.tolist()})
    write_csv('context_groups.csv', grouped_rows)
    write_csv('trend_scenarios_retrospective.csv', scenario_rows)

    # Resample natural weeks, preserving highly overlapping windows within weeks.
    daily = pd.DataFrame(daily_rows)
    daily['week'] = pd.to_datetime(daily['date']).dt.strftime('%G-W%V')
    weekly = daily.groupby(['week', 'target'])[['model_abs_sum', 'baseline_abs_sum']].sum()
    rng = np.random.default_rng(20261021)
    for row in target_rows:
        wv = weekly.xs(row['target'], level='target').to_numpy()
        draws = rng.integers(0, len(wv), size=(2000, len(wv)))
        bs = wv[draws].sum(axis=1)
        skill_bs = 1 - bs[:, 0] / bs[:, 1]
        row['weekly_block_skill_ci_low'], row['weekly_block_skill_ci_high'] = np.quantile(skill_bs, [.025, .975]).tolist()
        row['positive_skill_days'] = int((daily.loc[daily['target'] == row['target'], 'mae_skill'] > 0).sum())
    write_csv('target_diagnostics.csv', [{key: value for key, value in row.items() if not key.endswith('phases')} for row in target_rows])

    # Future-time-fixed traces do not fuse different forecast horizons.
    series = pd.DataFrame({'timestamp': pd.to_datetime(target_times)})
    for j, target in enumerate(targets):
        series['true_' + target] = y[:, -1, j]
        series['pred_' + target] = p[:, -1, j]
    series = series.set_index('timestamp').resample('10min').mean()
    series.to_parquet(OUT / 'fixed_10min_horizon_display.parquet')
    record = pd.DataFrame({'window_id': np.arange(n), 'chunk': chunk_number, 'target_end_time': pd.to_datetime(target_times), 'current_steam': current_load, 'past_control_range_score': past_actions, 'future_control_range_score_retrospective': future_actions, 'input_missing_fraction': input_missing_fraction})
    for j, target in enumerate(targets):
        record['mae_' + target] = win_mae[:, j]
        record['base_mae_' + target] = win_base[:, j]
    record.to_parquet(OUT / 'window_diagnostics.parquet', index=False)

    causal_rows = causal_audit(chunks, controls)
    write_csv('causal_preprocessing_audit.csv', causal_rows)
    source_quality_audit()
    supplement_analysis(meta, chunks, train_chunks, record, targets, mean, scale, inds, original_p, original_y, alpha, bias_cal)
    plot_results(targets, daily, target_rows, h_abs / n, h_base / n, grouped_rows, scenario_rows, series, candidate_rows, p, y, origin, target_times, worst_cases)
    meta_out = {'test_windows': n, 'test_chunks': len(chunks), 'unique_used_history_rows_chunkwise': used_history_rows, 'load_train_tertile_cuts': load_cuts.tolist(), 'controls_used': controls, 'past_action_calibration_cuts': cuts_past.tolist(), 'future_action_calibration_cuts': cuts_future.tolist(), 'diagnostic_calibration_chunks': 26, 'calibration_windows': int(cal.sum()), 'gap_chunk': chunks[26]['core_start'][:10], 'replication_windows': int(check.sum()), 'replication_start': chunks[27]['core_start'][:10], 'history_missing_windows': int((input_missing_fraction > 0).sum()), 'history_missing_window_fraction': float((input_missing_fraction > 0).mean()), 'original_mode_labels_available': False, 'warning': '探索性分析。校准和复核子集均来自此前已检查的测试期，不是原始验证集或全新最终测试。动作强弱由控制信号范围构造，不等同于操作日志或因果效果。'}
    final_state = {name: {'size': (ART / name).stat().st_size, 'mtime_ns': (ART / name).stat().st_mtime_ns} for name in source_state}
    if source_state != final_state:
        raise RuntimeError('original model artifacts changed during diagnostics')
    meta_out['original_artifacts_unchanged'] = True
    meta_out['original_artifact_stats'] = source_state
    (OUT / 'diagnostic_metadata.json').write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(meta_out, target_rows, daily, grouped_rows, scenario_rows, candidate_rows, causal_rows, coverage_rows)
    p._mmap.close()
    y._mmap.close()
    for cache_name in ['_target_pred.npy', '_target_true.npy']:
        (OUT / cache_name).unlink()
    print(json.dumps(meta_out, ensure_ascii=True), flush=True)
    print('DONE: ' + str(OUT), flush=True)


def supplement_analysis(meta, chunks, train_chunks, record, targets, mean, scale, inds, original_p, original_y, alpha, bias_cal):
    missing = pd.read_csv(OUT / 'input_missingness.csv')
    selected_codes = missing.loc[missing['missing_rate'] > .1, 'feature'].tolist()
    split_loads = {'test': record['current_steam'].to_numpy()}
    feature_rows, load_rows = [], []
    for split, chunk_list in [('train', train_chunks), ('val', csv_rows(DATA / 'splits/val_chunks.csv')), ('test', chunks)]:
        all_load = []
        missing_count = np.zeros(len(selected_codes), dtype='int64')
        windows_total = 0
        for c in chunk_list:
            frame = pd.read_parquet(DATA / 'frames' / Path(c['frame_path']).name, columns=[targets[0], *selected_codes])
            w = pd.read_csv(DATA / 'windows' / Path(c['window_path']).name, usecols=['input_end_pos'])
            values = frame.iloc[w['input_end_pos'].to_numpy(dtype=int)]
            all_load.append(values[targets[0]].to_numpy())
            missing_count += (~np.isfinite(values[selected_codes].to_numpy(dtype='float64'))).sum(axis=0)
            windows_total += len(values)
        loads = np.concatenate(all_load)
        split_loads[split] = loads
        load_rows.append({'split': split, 'windows': len(loads), 'mean_steam': float(loads.mean()), 'std_steam': float(loads.std()), 'p10': float(np.quantile(loads, .1)), 'median': float(np.median(loads)), 'p90': float(np.quantile(loads, .9))})
        for j, code in enumerate(selected_codes):
            feature_rows.append({'split': split, 'feature': code, 'forecast_origins': windows_total, 'missing_origins': int(missing_count[j]), 'missing_origin_rate': float(missing_count[j] / windows_total)})
        print('distribution audit: ' + split, flush=True)
    write_csv('load_distribution_by_split.csv', load_rows)
    write_csv('persistent_missing_inputs_by_split.csv', feature_rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(35, 70, 100)
    for split, label in [('train', '训练期'), ('val', '原验证期'), ('test', '测试期')]:
        ax.hist(split_loads[split], bins=bins, density=True, histtype='step', linewidth=1.8, label=label)
    ax.set(xlabel='预测起点主蒸汽流量（t/h）', ylabel='概率密度', title='同一特征在训练、验证和测试期的分布是否一致')
    ax.legend(frameon=False)
    save_fig('fig09_load_distribution', fig)

    # Overlapping forecast windows are never treated as independent experiments here.
    end_true = np.asarray(original_y[:, -1, :][:, inds], dtype='float64') * scale[inds] + mean[inds]
    end_pred = np.asarray(original_p[:, -1, :][:, inds], dtype='float64') * scale[inds] + mean[inds]
    end_origin = np.empty_like(end_true)
    cursor = 0
    rule_daily = []
    for day, c in enumerate(chunks):
        frame, w = load_chunk(c, targets)
        count = len(w)
        o = frame.iloc[w['input_end_pos'].to_numpy(dtype=int)][targets].to_numpy(dtype='float64')
        end_origin[cursor:cursor + count] = o
        if day >= 27:
            sums = {method: np.zeros(len(targets)) for method in ['原模型', '分时距组合', '分时距偏差校正', '预测输出3点平滑']}
            for local in range(0, count, 256):
                end = min(local + 256, count)
                b = slice(cursor + local, cursor + end)
                pp = np.asarray(original_p[b][:, :, inds], dtype='float64') * scale[inds] + mean[inds]
                yy = np.asarray(original_y[b][:, :, inds], dtype='float64') * scale[inds] + mean[inds]
                oo = o[local:end, None, :]
                blend, corrected = pp.copy(), pp.copy()
                for gi, hs in enumerate([slice(0, 12), slice(12, 30), slice(30, 60)]):
                    blend[:, hs] = oo + alpha[gi] * (pp[:, hs] - oo)
                    corrected[:, hs] -= bias_cal[gi]
                padded = np.concatenate([oo, oo, pp], axis=1)
                smooth = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3
                for method, v in [('原模型', pp), ('分时距组合', blend), ('分时距偏差校正', corrected), ('预测输出3点平滑', smooth)]:
                    sums[method] += np.abs(v - yy).sum(axis=(0, 1))
            for method, ss in sums.items():
                for j, target in enumerate(targets):
                    rule_daily.append({'date': c['core_start'][:10], 'target': target, 'name': NAMES[j], 'method': method, 'windows': count, 'mae': ss[j] / (count * 60), 'improvement_vs_original': 1 - ss[j] / sums['原模型'][j]})
        cursor += count
    write_csv('exploratory_replication_by_day.csv', rule_daily)
    matrices = {'true_endpoint_level': np.corrcoef(end_true.T), 'true_future_change': np.corrcoef((end_true - end_origin).T), 'endpoint_error': np.corrcoef((end_pred - end_true).T)}
    pair_rows = []
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=targets, columns=targets).to_csv(OUT / (name + '_correlation.csv'), encoding='utf-8-sig')
        for j in range(len(targets)):
            for kk in range(j + 1, len(targets)):
                pair_rows.append({'quantity': name, 'target_a': targets[j], 'target_b': targets[kk], 'correlation': matrix[j, kk]})
    write_csv('target_pair_correlations.csv', pair_rows)
    coef = np.polyfit(end_true[:, 1], end_true[:, 2], 1)
    pred_oxygen_b = np.polyval(coef, end_true[:, 1])
    oxygen = {'slope_B_vs_A': coef[0], 'intercept_B_vs_A': coef[1], 'linear_relation_rmse': float(np.sqrt(np.square(pred_oxygen_b - end_true[:, 2]).mean()))}
    timestamps = record['target_end_time'].to_numpy(dtype='datetime64[ns]')
    grid_start = timestamps.min() - np.timedelta64(590, 's')
    end_idx = ((timestamps - grid_start) / np.timedelta64(10, 's')).astype(int)
    footprint = np.zeros(int(end_idx.max()) + 2, dtype='int32')
    np.add.at(footprint, end_idx - 59, 1)
    np.add.at(footprint, end_idx + 1, -1)
    covered = int((footprint[:-1].cumsum() > 0).sum())
    coverage = {'unique_forecast_origins': len(np.unique(timestamps)), 'origin_hours': len(np.unique(timestamps)) / 360, 'globally_unique_future_label_hours': covered / 360, 'label_coverage_span_hours': len(footprint[:-1]) / 360, 'label_coverage_fraction_in_span': covered / len(footprint[:-1]), 'oxygen_relation': oxygen, 'missing_codes_audited': selected_codes}
    (OUT / 'supplementary_summary.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name, title in zip(axes, ['true_future_change', 'endpoint_error'], ['未来10分钟净变化的相关性', '10分钟预测误差的相关性']):
        im = ax.imshow(matrices[name], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(targets)), [s.replace('B3', '') for s in targets], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(targets)), NAMES, fontsize=8)
        ax.set_title(title)
    axes[1].set_yticklabels([])
    fig.colorbar(im, ax=axes.tolist(), label='相关系数', shrink=.8)
    save_fig('fig10_target_coupling', fig)


def causal_audit(chunks, controls):
    path = ROOT / 'offline/forecast/data_engineering/build_model_datasets.py'
    spec = importlib.util.spec_from_file_location('diagnostic_data_builder', path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    cfg = json.loads((path.parent / 'config/default_config.json').read_text(encoding='utf-8'))
    audit_codes = list(dict.fromkeys(['B3FT81', 'B3FT82', 'B3FT83', 'B3HS15GT', 'B3ZI81', *controls[:3]]))
    result, examples = [], []
    for day in [0, len(chunks) // 2, len(chunks) - 1]:
        c = chunks[day]
        frame, windows = load_chunk(c, ['timestamp', *audit_codes])
        times = pd.DatetimeIndex(frame['timestamp'])
        positions = windows['input_end_pos'].to_numpy(dtype=int)
        for code in audit_codes:
            file = CACHE / (code + '.parquet')
            raw = pd.read_parquet(file, columns=['timestamp', code], filters=[('timestamp', '>=', times[0].to_pydatetime()), ('timestamp', '<=', times[-1].to_pydatetime())]).set_index('timestamp')[code].reindex(times).reset_index(drop=True)
            cleaned, _ = builder.isolated_input_spike_clean_series(pd, raw, cfg['input_spike_filter'])
            full = cleaned.interpolate(method='linear', limit=cfg['input_interpolate_limit'], limit_direction='both').ffill(limit=cfg['input_forward_fill_limit'])
            # Inspect ordinary origins and origins immediately affected by raw/cleaned gaps.
            events = positions[raw.iloc[positions].isna().to_numpy() | cleaned.iloc[positions].isna().to_numpy()]
            chosen = np.unique(np.r_[positions[np.linspace(0, len(positions) - 1, 12, dtype=int)], events[:12]])
            count_changed, count_matched, changed_matched, max_change = 0, 0, 0, 0.
            for end in chosen:
                prefix, _ = builder.isolated_input_spike_clean_series(pd, raw.iloc[:end + 1], cfg['input_spike_filter'])
                prefix = prefix.interpolate(method='linear', limit=cfg['input_interpolate_limit'], limit_direction='both').ffill(limit=cfg['input_forward_fill_limit'])
                last = slice(max(0, int(end) - 14), int(end) + 1)
                fv, pv = full.iloc[last].to_numpy(), prefix.iloc[last].to_numpy()
                stored = frame[code].iloc[last].to_numpy()
                matched = bool(np.allclose(stored, fv, rtol=2e-5, atol=2e-4, equal_nan=True))
                changed = not np.allclose(fv, pv, rtol=2e-5, atol=2e-4, equal_nan=True)
                count_matched += matched
                count_changed += changed
                changed_matched += changed and matched
                if changed and matched:
                    affected = np.flatnonzero(~np.isclose(fv, pv, rtol=2e-5, atol=2e-4, equal_nan=True))
                    later = raw.iloc[int(end) + 1:].first_valid_index()
                    for at in affected:
                        index = int(last.start) + int(at)
                        examples.append({'date': c['core_start'][:10], 'input': code, 'forecast_origin_time': str(times[int(end)]), 'affected_history_time': str(times[index]), 'saved_value': float(stored[at]), 'full_chunk_clean_value': float(fv[at]), 'history_prefix_clean_value': float(pv[at]), 'raw_value_at_affected_time': float(raw.iloc[index]), 'next_observation_after_origin_time': str(times[int(later)]) if later is not None else '', 'next_raw_value': float(raw.iloc[int(later)]) if later is not None else np.nan})
                diff = np.abs(fv - pv)
                if np.isfinite(diff).any():
                    max_change = max(max_change, float(np.nanmax(diff)))
            result.append({'date': c['core_start'][:10], 'input': code, 'origins_tested': len(chosen), 'full_clean_matches_saved': count_matched, 'full_vs_prefix_diff_origins': count_changed, 'different_and_matches_saved': changed_matched, 'max_abs_prefix_difference': max_change, 'sample_selection': '12 evenly spaced + up to12 missing-related; not prevalence estimate'})
    write_csv('causal_preprocessing_examples.csv', examples)
    return result


def source_quality_audit():
    import pyarrow.parquet as pq

    codes = ['B3_F02B_MV', 'B3_F03A_MV', 'B3_F03B_MV', 'B3_F04A_MV', 'B3QT04C', 'B3QT05', 'B3FT90', 'B3HV84GT', 'B3ZI03B']
    point_map = pd.read_csv(ROOT / 'assets/profiles/xingrong_3/points/point_map.csv').fillna('').set_index('point_code')
    rows = []
    for code in codes:
        raw = pd.read_parquet(CACHE / (code + '.parquet'), columns=['timestamp', code])
        valid = raw.loc[np.isfinite(raw[code]), ['timestamp', code]]
        test = valid.loc[valid['timestamp'] >= pd.Timestamp('2026-03-08')]
        source = ROOT / 'assets/data/xingrong_3/silver/by_point' / ('point_code=' + code) / 'data.parquet'
        test_source = pd.read_parquet(source, columns=['timestamp', 'value'], filters=[('timestamp', '>=', pd.Timestamp('2026-03-08').to_pydatetime())])
        source_meta = pq.ParquetFile(source).metadata
        source_maxima = [source_meta.row_group(i).column(0).statistics.max for i in range(source_meta.num_row_groups) if source_meta.row_group(i).column(0).statistics is not None]
        rows.append({'input': code, 'name': point_map.loc[code, 'cn_name'], 'unit': point_map.loc[code, 'unit'], 'cache_first_finite': str(valid['timestamp'].min()), 'cache_last_finite': str(valid['timestamp'].max()), 'silver_max_timestamp': str(max(source_maxima)) if source_maxima else '', 'silver_test_rows': len(test_source), 'silver_test_finite_rows': int(np.isfinite(test_source['value']).sum()), 'cache_test_rows': len(test), 'cache_test_negative_fraction': float((test[code] < 0).mean()) if len(test) else np.nan, 'cache_test_median': float(test[code].median()) if len(test) else np.nan})
    write_csv('missing_input_source_audit.csv', rows)


def plot_results(targets, daily, target_rows, ha, hb, grouped_rows, scenarios, series, candidates, p, y, origin, times, worst):
    fig, axes = plt.subplots(7, 2, figsize=(15, 22))
    for j, ax in enumerate(axes.ravel()):
        ax.plot(np.arange(1, 61) / 6, ha[:, j], color='#0077BB', label='原模型')
        ax.plot(np.arange(1, 61) / 6, hb[:, j], color='#EE7733', ls='--', label='保持当前值')
        ax.set(title=NAMES[j], xlabel='预测时距（分钟）', ylabel='MAE (' + UNITS[j] + ')')
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('误差随预测时距的变化：与保持当前值逐点比较', y=.999)
    fig.tight_layout(rect=[0, 0, 1, .985])
    save_fig('fig01_horizon_errors', fig)

    values = daily.pivot(index='target', columns='date', values='mae_skill').reindex(targets)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(values.to_numpy(), aspect='auto', cmap='RdBu', vmin=-.5, vmax=.5)
    ax.set_yticks(np.arange(14), NAMES)
    labels = values.columns.tolist()
    ax.set_xticks(np.arange(0, len(labels), 4), [s[5:] for s in labels[::4]], rotation=30)
    ax.set(title='逐日相对保持当前值的MAE改善：蓝色改善，红色退化', xlabel='测试数据块起始日期')
    fig.colorbar(im, ax=ax, label='1 − 模型MAE / 持续值MAE', extend='both')
    save_fig('fig02_daily_skill', fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for side, field in enumerate(['pred_patch_phases', 'true_patch_phases']):
        v = np.array([row[field] for row in target_rows])
        v = v / v.mean(axis=1, keepdims=True)
        im = axes[side].imshow(v, aspect='auto', cmap='Blues', vmin=0, vmax=2.5)
        axes[side].set_yticks(np.arange(14), NAMES)
        axes[side].set_xticks(np.arange(8), np.arange(1, 9))
        axes[side].set(xlabel='8点输出块中的步差相位（第4列为衔接处）', title=['预测曲线的相位波动', '真实标签的相位波动'][side])
    axes[1].set_yticklabels([])
    fig.colorbar(im, ax=axes.tolist(), label='相邻点变化量 / 该目标平均变化量', shrink=.8)
    save_fig('fig03_patch_phase', fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, gt, labels in zip(axes, ['load', 'future_control_range_retrospective'], [['低负荷', '中负荷', '高负荷'], ['小变化', '中变化', '大变化']]):
        data = pd.DataFrame([r for r in grouped_rows if r['group_type'] == gt]).pivot(index='target', columns='group', values='mae_skill').reindex(targets)
        im = ax.imshow(data.to_numpy(), aspect='auto', cmap='RdBu', vmin=-.5, vmax=.5)
        ax.set_yticks(np.arange(14), NAMES)
        ax.set_xticks(np.arange(len(data.columns)), labels[:len(data.columns)])
        ax.set_title('按当前蒸汽量分组' if gt == 'load' else '按随后控制信号变化分组（回顾性）')
    axes[1].set_yticklabels([])
    fig.colorbar(im, ax=axes.tolist(), label='相对持续值MAE改善', shrink=.8, extend='both')
    save_fig('fig04_operating_context', fig)

    scen = pd.DataFrame(scenarios).pivot(index='target', columns='scenario', values='mae_skill').reindex(targets)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(scen.to_numpy(), aspect='auto', cmap='RdBu', vmin=-.5, vmax=.5)
    ax.set_yticks(np.arange(14), NAMES)
    ax.set_xticks(np.arange(len(scen.columns)), scen.columns)
    ax.set_title('趋势延续、反转与新变化：场景由历史和未来真值回顾性定义')
    fig.colorbar(im, ax=ax, label='相对持续值MAE改善', extend='both')
    save_fig('fig05_trend_scenarios', fig)

    fig, axes = plt.subplots(7, 2, figsize=(17, 21))
    for j, ax in enumerate(axes.ravel()):
        ax.plot(series.index, series['true_' + targets[j]], color='#0077BB', lw=.8, label='实际')
        ax.plot(series.index, series['pred_' + targets[j]], color='#EE7733', lw=.8, alpha=.8, label='提前10分钟预测')
        ax.set(title=NAMES[j], ylabel=UNITS[j])
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('固定10分钟预测时距的全测试期走势（仅显示采用10分钟分桶均值；无跨时距融合）', y=.998)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, .985])
    save_fig('fig06_fixed_horizon_long_trends', fig)

    cand = pd.DataFrame(candidates)
    base = cand[cand['group'] == '原模型'].set_index('target')['mae']
    fig, ax = plt.subplots(figsize=(11, 6))
    methods = ['分时距组合', '分时距偏差校正', '预测输出3点平滑']
    for idx, method in enumerate(methods):
        v = cand[cand['group'] == method].set_index('target')['mae'].reindex(targets)
        improve = 100 * (1 - v / base.reindex(targets))
        ax.barh(np.arange(14) + (idx - 1) * .23, improve, height=.22, label=method)
    ax.axvline(0, color='black', lw=.8)
    ax.set_yticks(np.arange(14), NAMES)
    ax.set(xlabel='相对原模型MAE改善（%；负值为变差）', title='探索性后处理：较早测试日确定参数，较晚测试日复核')
    ax.legend(frameon=False)
    save_fig('fig07_exploratory_replication', fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (j, (score, index)) in zip(axes, worst.items()):
        x = np.r_[0, np.arange(1, 61) / 6]
        ax.plot(x, np.r_[origin[index, j], y[index, :, j]], color='#0077BB', label='实际')
        ax.plot(x, np.r_[origin[index, j], p[index, :, j]], color='#EE7733', label='原模型')
        ax.axhline(origin[index, j], color='gray', ls='--', label='保持当前值')
        ax.set(title=f'{NAMES[j]}：最大窗口MAE案例 {str(times[index])[:19]}（窗口终点）', xlabel='预测后分钟', ylabel=UNITS[j])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_fig('fig08_worst_cases', fig)


def write_report(meta, target_rows, daily, context, scenarios, candidates, causal, coverage):
    rows = []
    rows += ['# 模型1：无需重训的深入诊断', '', '本报告只读取现有模型输出、测试帧、点位映射和重采样缓存。未修改权重、训练数据或运行时策略。', '', '## 分析范围与口径', '', f'- 全量：{meta["test_windows"]:,}个测试窗口，{meta["test_chunks"]}个数据块，14个目标、60个预测时距。每块首/中/末窗口的60步真值均与本地帧核对。', '- 时间相关性：逐日比较，并按自然周重抽样计算相对持续值改进区间；窗口不是独立样本。', '- 真实值口径：当前清洗后的训练帧标签，不等同于未经处理的仪表原值。', '- 两路氧量名称以A/B中性命名；现场测点位置仍需以正式点表确认。', '- 自动/手动、吹灰、投料事件标签未在本次可用数据中形成可验证事件表；控制信号范围只能用于操作变化代理分组，不等同于模式或事件日志。', '', '## 逐目标结果', '', '|目标|全程MAE|相对持续值改进|改善日期数/52|周块95%区间|窗口偏移占MSE比例|', '|---|---:|---:|---:|---:|---:|']
    for r in target_rows:
        rows.append(f'|{r["name"]}|{r["mae"]:.4g} {r["unit"]}|{r["mae_skill"]:.1%}|{r["positive_skill_days"]}/52|[{r["weekly_block_skill_ci_low"]:.1%}, {r["weekly_block_skill_ci_high"]:.1%}]|{r["window_offset_mse_share"]:.1%}|')
    rows += ['', '“窗口偏移占MSE比例”把每个10分钟窗口内的平均预测偏移从误差中分解出来。它利用未来真值，只用于定位误差来源；不能视为可部署校正的收益。周块区间只描述当前测试期的时间变动，不覆盖其他季节和新运行策略。', '', '## 误差随时距、负荷与场景的变化', '', '- `horizon_metrics.csv`和图01逐10秒比较模型与持续值，避免只看全程平均分数。', f'- 当前蒸汽量按训练期三分位数分组，阈值为{meta["load_train_tertile_cuts"][0]:.3f}、{meta["load_train_tertile_cuts"][1]:.3f} t/h。', '- 控制变化得分为控制点窗口范围除以各自训练标准差后的均值；历史组使用过去5分钟，未来组使用随后10分钟。后者是回顾性分析，不能实时作为策略选择输入。', '- 趋势场景以过去5分钟净变化和未来10分钟净变化分类，显著变化门槛为各目标训练标准差的0.1倍。场景不是现场事件标签。', '', '|目标|趋势延续改进|趋势反转改进|新出现变化改进|', '|---|---:|---:|---:|']
    for target in target_rows:
        vals = {r['scenario']: r['mae_skill'] for r in scenarios if r['target'] == target['target']}
        rows.append(f'|{target["name"]}|{vals.get("趋势延续", np.nan):.1%}|{vals.get("趋势反转", np.nan):.1%}|{vals.get("新出现变化", np.nan):.1%}|')
    rows += ['', '## 输出分块伪波动', '', '8点分块下，考虑64点补齐后截取60点，衔接处对应步差相位4。图03在预测和真实标签上使用同一相位比较。', '', '|目标|预测边界/其他位置变化量|真实边界/其他位置变化量|预测/真实累计变化量|', '|---|---:|---:|---:|']
    for r in target_rows:
        rows.append(f'|{r["name"]}|{r["pred_boundary_jump_ratio"]:.3f}|{r["true_boundary_jump_ratio"]:.3f}|{r["pred_total_variation_ratio"]:.3f}|')
    rows += ['', '边界效应是结构性伪波动的证据，但不能据此推断它解释全部MAE。总变差是相邻点绝对变化之和，不是方差或振幅。', '', '## 无需重训的后处理探索', '', f'在前26个测试块（{meta["calibration_windows"]:,}窗）估计简单规则，隔开{meta["gap_chunk"]}一个数据块，再在从{meta["replication_start"]}开始的{meta["replication_windows"]:,}个较晚窗口复核。', '', '**两个子集都来自此前已检查的测试期，不是原始验证集或全新最终测试。以下结果只能用于探索和确定下一步实验，不能作为正式上线效果承诺。**', '', '- 分时距组合：在10–120、130–300、310–600秒三段各自求一个[0,1]内的模型权重，以较早子集平方误差最小为目标。', '- 偏差校正：每目标每时距段估计一个平均偏移。', '- 3点输出平滑：仅使用预测序列及当前观测，不使用未来真值；属于固定规则。', '', '|目标|原模型MAE|组合MAE|偏差校正MAE|输出平滑MAE|', '|---|---:|---:|---:|---:|']
    candidates = pd.DataFrame(candidates)
    for r in target_rows:
        v = candidates[candidates['target'] == r['target']].set_index('group')['mae']
        rows.append(f'|{r["name"]}|{v["原模型"]:.4g}|{v["分时距组合"]:.4g}|{v["分时距偏差校正"]:.4g}|{v["预测输出3点平滑"]:.4g}|')
    rows += ['', '## 输入缺失与时间因果清洗核验', '', f'- 在已使用历史窗中，{meta["history_missing_windows"]:,}个窗口（{meta["history_missing_window_fraction"]:.1%}）仍含至少一个非有限输入帧值。桥接缩放器会把它映射为标准化0，即训练均值；“训练不再NaN”不等于原始输入全部有效。逐列统计见input_missingness.csv。', '- 对首/中/末3个测试块的8个输入，比较完整块清洗与“只给到预测起点”的前缀清洗，检查历史末端15点。每组包含12个均匀起点，及最多12个缺失相关起点，因此不能用检出比例估计全数据泄漏率。', '- 只有完整块清洗能复现已保存帧、且前缀清洗不同的起点，才构成当前样本受未来信息影响的直接证据；其他差异仅表明当前清洗算法具有前视依赖。', '', f'- 测试组合数：{len(causal)}；总检查起点：{sum(r["origins_tested"] for r in causal)}；完整清洗复现保存帧的起点：{sum(r["full_clean_matches_saved"] for r in causal)}；其中前缀不同：{sum(r["different_and_matches_saved"] for r in causal)}。', '', '## 图与数据索引', '', '- 图01：14个目标的逐时距MAE及持续值比较。', '- 图02：逐日收益热图；关注退化日期是否集中。', '- 图03：8点分块的相位波动。', '- 图04：负荷与随后控制变化分组。', '- 图05：趋势延续/反转等场景。', '- 图06：固定提前10分钟的全测试期曲线，显示采用10分钟分桶平均，未混合其他预测时距；缺测桶保留空白。', '- 图07：探索性后处理在较晚日期的复核。', '- 图08：最大窗口MAE案例，不代表典型工况。', '- 所有PNG另有PDF；汇总CSV、逐窗Parquet和元数据用于复算。', '', '## 使用限制', '', '这是同一批已观察测试结果上的诊断研究。分组相关性不能解释为控制动作的因果效果，后处理不能直接上线；任何选出的规则应在原始验证期或新的时间段独立验收。未启动新训练，未修改模型或数据集。']
    enrich_report(rows, meta, context, candidates, causal, target_rows)
    (OUT / 'analysis_report.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def enrich_report(rows, meta, context, candidates, causal, target_rows):
    supplementary = json.loads((OUT / 'supplementary_summary.json').read_text(encoding='utf-8'))
    distributions = pd.read_csv(OUT / 'load_distribution_by_split.csv')
    candidate_table = pd.DataFrame(candidates).pivot(index='target', columns='group', values='mae')
    load_low = next(r['windows'] for r in context if r['group_type'] == 'load' and r['group'] == 0 and r['target'] == target_rows[0]['target']) / meta['test_windows']
    temp = candidate_table.loc['B3TE03']
    brief = ['', '## 主要发现及其含义', '',
        '1. **优势稳定存在于部分过程目标。** 蒸汽流量51/52日优于持续值；水位、主蒸汽温度和负压52/52日均优于持续值。不能因负压曲线偏平就判定它毫无预测价值。',
        '2. **一次风预热温度的短时预测需要保护。** 它仅20/52日优于持续值；按前后净变化定义的“变化较小”场景占约74%，该场景误差反而高于持续值。该分类不等于整个窗口内始终平稳。',
        f'3. **测试覆盖存在明显负荷偏移。** {load_low:.1%}测试窗口落在训练期有效帧蒸汽量的低三分位区间；本批数据无法充分证明中高负荷能力。训练/验证/测试预测起点分布见图09。',
        '4. **反转并非所有目标的共同短板。** 蒸汽流量对趋势延续的增益高于反转，但汽包水位对反转仍有明显增益。这支持按过程分别诊断，而不是统一增加趋势约束。',
        '5. **输出块衔接规律已全量复核。** 预测在固定相位集中跳变，真实标签基本没有同相位变化；固定3点输出平滑的MAE收益小于视觉改善，不能将伪波动当作全部误差来源。',
        f'6. **存在便宜但需独立验证的改善线索。** 在较晚89,758窗，预热温度分时距组合MAE从{temp["原模型"]:.3f}降到{temp["分时距组合"]:.3f}°C，改善{1 - temp["分时距组合"] / temp["原模型"]:.1%}。两个子集均属已观察测试期，不是正式独立验收。',
        '7. **缺失输入与有效输出必须分开。** 某些辅助输入长期缺失，桥接器以训练均值补齐；不应因此删除所有多目标窗口，但实际有效输入数和缺失状态需要审阅。',
        f'8. **发现当前保存样本的前视清洗实例。** 缺失相关抽查中有{sum(r["different_and_matches_saved"] for r in causal)}个“完整块清洗能复现保存帧、仅历史前缀清洗不同”的输入实例。不能据此推算全测试泄漏率或精度高估幅度。', '']
    rows[1:1] = brief
    rows += ['', '## 补充：训练、验证与测试分布', '', '|数据期|预测起点数|蒸汽量均值|中位数|10%分位|90%分位|', '|---|---:|---:|---:|---:|---:|']
    for _, r in distributions.iterrows():
        rows.append(f'|{r["split"]}|{int(r["windows"]):,}|{r["mean_steam"]:.3f}|{r["median"]:.3f}|{r["p10"]:.3f}|{r["p90"]:.3f}|')
    rows += ['', '负荷分布不同是事实；究竟来自负荷策略、季节、设备状态还是标定变化，当前数据不能确定。三分位是相对训练数据的分组，不代表厂区额定负荷等级。', '', '## 补充：长期缺失的输入', '', '下表是预测起点上的缺失比例，三期使用同一统计方式；与input_missingness.csv的唯一历史帧比例分母不同。', '', '|输入|训练缺失率|原验证期缺失率|测试缺失率|', '|---|---:|---:|---:|']
    missing = pd.read_csv(OUT / 'persistent_missing_inputs_by_split.csv').pivot(index='feature', columns='split', values='missing_origin_rate')
    for code, r in missing.iterrows():
        rows.append(f'|{code}|{r["train"]:.1%}|{r["val"]:.1%}|{r["test"]:.1%}|')
    source_audit = pd.read_csv(OUT / 'missing_input_source_audit.csv')
    rows += ['', '### 继续追溯到原始Silver与重采样缓存', '', '|输入|含义|原始Silver最后时间|测试期Silver有效行|测试缓存负值比例|测试缓存中位数|', '|---|---|---|---:|---:|---:|']
    for _, r in source_audit.iterrows():
        negative = f'{r["cache_test_negative_fraction"]:.1%}' if np.isfinite(r['cache_test_negative_fraction']) else '无数据'
        median = f'{r["cache_test_median"]:.5g}' if np.isfinite(r['cache_test_median']) else '无数据'
        rows.append(f'|{r["input"]}|{r["name"]}|{r["silver_max_timestamp"]}|{int(r["silver_test_finite_rows"]):,}|{negative}|{median}|')
    rows += ['', '四个配风指令的原始Silver止于2025-11-11；烟气H₂O、另一O₂输入止于2026-02-11。这些测试期空值不是本次清洗凭空造成，也不能仅用重新训练来恢复信息。2026-10-22现场确认：属于测点问题，无法通过数据层面恢复。本研究将其视为不可用输入，不再把补档、改名排查或插值恢复作为当前改进方向。加载时填训练均值仅用于兼容既有模型，不代表恢复了测量信息。', '', '两路阀门输入原始值仍有覆盖，中位数为−0.07326%和−0.02442%；当前百分比硬范围0–100将这些负值筛除。2026-10-22现场确认：这些负值代表有问题的信号，现有筛除正确。维持筛除，不扩大负值容差，也不将其恢复为有效输入。数值接近零本身不足以推翻现场信号质量判断。', '', '前视清洗的具体时间与前后数值记录在causal_preprocessing_examples.csv。例如2026-03-08 06:34:40的一级减温水流量，原始/仅历史清洗值为2.462154 t/h，完整块清洗及保存值为2.426231 t/h，下一条观测发生在06:34:50。该例证明当时不可见的后续样本参与了历史输入清洗，但不量化它对最终预测精度的影响。']
    rows += ['', '训练和测试都空的列没有真实过程信息；训练可用而测试缺失属于可用性变化。不能只把NaN转0就称为质量已解决，也不应为少数辅助点缺失删除所有窗口。', '', '## 补充：目标耦合与信息冗余', '', '使用全量测试的10分钟终点；变化相对各窗当前值。相关可来自共同过程、标定或计算关系，不足以证明两个标签是重复传感器。']
    pairs = pd.read_csv(OUT / 'target_pair_correlations.csv')
    for a, b in [('B3QT00', 'B3_O2_S1'), ('B3_DRUMLT1', 'B3_DRUMLT2'), ('B3PT81A', 'B3PT81B'), ('B3PT04A', 'B3PT04B')]:
        found = pairs[(pairs['target_a'] == a) & (pairs['target_b'] == b)].set_index('quantity')['correlation']
        rows.append(f'- {a} / {b}：数值相关{found["true_endpoint_level"]:.6f}，未来净变化相关{found["true_future_change"]:.6f}，预测误差相关{found["endpoint_error"]:.6f}。')
    rows += ['', f'两路氧量线性拟合残差RMSE约{supplementary["oxygen_relation"]["linear_relation_rmse"]:.6g}个百分点；是否共用测量源或存在换算关系，需要核对DCS信号定义。', '', '## 补充：后处理逐日复核', '', '|目标|组合改善日期数/25|偏差校正改善日期数/25|输出平滑改善日期数/25|', '|---|---:|---:|---:|']
    replicate_days = pd.read_csv(OUT / 'exploratory_replication_by_day.csv')
    for r in target_rows:
        selected = replicate_days[replicate_days['target'] == r['target']]
        counts = [int((selected.loc[selected['method'] == method, 'improvement_vs_original'] > 0).sum()) for method in ['分时距组合', '分时距偏差校正', '预测输出3点平滑']]
        rows.append(f'|{r["name"]}|{counts[0]}/25|{counts[1]}/25|{counts[2]}/25|')
    rows += ['', '## 补充：实际时间覆盖及新图', '', f'测试窗口对应{supplementary["origin_hours"]:.1f}小时的独立起点网格。重叠未来标签按实际时间戳去重覆盖{supplementary["globally_unique_future_label_hours"]:.1f}小时，占首末标签间跨度的{supplementary["label_coverage_fraction_in_span"]:.1%}。20万个重叠窗不能当作20万个独立工况；也不能将跨越约两个月写成该时期全时段经过评估。', '', '- 图09：训练/原验证/测试的起点负荷分布。', '- 图10：目标变化与预测误差耦合。', '- 生成时约1.4GB目标缓存会在完成后移除，只保留必要报告数据。', '', '优先建议：审阅长期空输入和氧量信号定义；量化严格因果清洗后的实际预测差异；用独立日期验收预热温度短时保护及分时距组合；补齐不同负荷的验收。块衔接伪波动的视觉改善和MAE收益需分别验收。']


if __name__ == '__main__':
    main()
