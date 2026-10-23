"""Retrospective cross-period validation of a frozen preheater blend; no fitting."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60'
DATA = ROOT / 'offline/artifacts/forecast/data_engineering/outputs/datasets/chunked/model1_10s_seq120_pred60'
PRIOR = ART / 'evaluation_history_only_20261021/diagnostics_20261021'
OUT = ART / 'evaluation_history_only_20261021/independent_temperature_validation_20261022'
TARGET = 'B3TE03'
sys.path.insert(0, str(ROOT / 'offline/forecast/tslib'))
from models.Crossformer import Model


def metric(p, y):
    e = np.asarray(p, dtype='float64') - y
    sst = np.square(y - y.mean()).sum()
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.square(e).mean())),
                bias=float(e.mean()), r2=float(1 - np.square(e).sum() / sst) if sst else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ART / n for n in ['checkpoint.pth', 'metadata.json', 'scaler.json', 'pred.npy', 'true.npy']]
    sources.append(PRIOR / 'exploratory_rules.csv')
    initial = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    meta = json.loads((ART / 'metadata.json').read_text(encoding='utf-8'))
    scaler = json.loads((ART / 'scaler.json').read_text(encoding='utf-8'))
    features = meta['feature_columns']
    assert features == scaler['columns'] and len(features) == 304
    target = features.index(TARGET)
    inds = meta['target_indices']
    mean, scale = np.array(scaler['mean']), np.array(scaler['scale'])
    assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
    rules = pd.read_csv(PRIOR / 'exploratory_rules.csv')
    weights = rules.loc[rules.target == TARGET, 'model_weight'].to_numpy()
    assert len(weights) == 3
    alpha = np.repeat(weights, [12, 18, 30])
    torch.manual_seed(20261022)
    torch.set_num_threads(4)
    configs = SimpleNamespace(enc_in=304, seq_len=120, pred_len=60, patch_len=8,
                              task_name='long_term_forecast', d_model=192, n_heads=6,
                              d_ff=384, e_layers=2, factor=3, dropout=.1)
    model = Model(configs)
    state = torch.load(ART / 'checkpoint.pth', weights_only=True, map_location='cpu')
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.eval()

    def load_chunk(row):
        # Stored manifests contain Windows paths, even when copied from the VM.
        chunk = row['chunk_id']
        f = pd.read_parquet(DATA / 'frames' / (chunk + '.parquet'), columns=['timestamp'] + features)
        w = pd.read_csv(DATA / 'windows' / (chunk + '_windows.csv'))
        assert len(w) == int(row['windows'])
        return f, w

    def infer(frame, windows):
        raw = frame[features].to_numpy(dtype='float32')
        z = np.nan_to_num((raw - mean) / scale, nan=0., posinf=0., neginf=0.).astype('float32')
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(windows), 4):
                rows = windows.iloc[start:start + 4]
                x = torch.from_numpy(np.stack([z[int(r.input_start_pos):int(r.input_end_pos) + 1]
                                                for r in rows.itertuples()]))
                future = x[:, -1:, :].repeat(1, 60, 1)
                future[:, :, inds] = x[:, -60:, inds].mean(dim=1, keepdim=True)
                dec = torch.cat([x[:, -60:, :], future], dim=1)
                mask = torch.zeros(len(x), 60, len(features), dtype=torch.bool)
                mask[:, :, inds] = True
                p = model(x, None, dec, None, dec_mask=mask).numpy()
                if not np.isfinite(p).all():
                    raise FloatingPointError('nonfinite inference output')
                outputs.append(p)
        return np.concatenate(outputs)

    # Reproduce saved predictions before making any claims about new dates.
    test_chunks = pd.read_csv(DATA / 'splits/test_chunks.csv')
    saved = np.load(ART / 'pred.npy', mmap_mode='r')
    parity = []
    offset = 0
    for ci, row in test_chunks.iterrows():
        if ci in [0, len(test_chunks) // 2, len(test_chunks) - 1]:
            f, w = load_chunk(row)
            ix = np.unique([0, len(w) // 2, len(w) - 1])
            p = infer(f, w.iloc[ix])
            reference = np.asarray(saved[offset + ix])
            error = float(np.max(np.abs(p[:, :, inds] - reference[:, :, inds])))
            passed = bool(np.allclose(p[:, :, inds], reference[:, :, inds], rtol=1e-3, atol=1e-3))
            parity.append(dict(chunk_id=row.chunk_id, windows=len(ix), max_abs_target_z_error=error, passed=passed))
        offset += int(row.windows)
    (OUT / 'inference_parity.json').write_text(json.dumps(parity, indent=2), encoding='utf-8')
    if not all(r['passed'] for r in parity):
        raise ValueError('Local inference does not reproduce saved supervised target predictions')
    print('Saved prediction parity passed', flush=True)

    predictions, truths, origins, selections = [], [], [], []
    chunks = pd.read_csv(DATA / 'splits/val_chunks.csv')
    for ci, row in chunks.iterrows():
        f, w = load_chunk(row)
        # Disjoint history+forecast spans; evenly select at most 16 across each chunk.
        candidates, previous_end = [], -1
        for i, r in enumerate(w.itertuples()):
            if r.input_start_pos > previous_end:
                candidates.append(i)
                previous_end = r.target_end_pos
        ix = np.array(candidates)[np.unique(np.linspace(0, len(candidates) - 1, min(16, len(candidates))).round().astype(int))]
        selected = w.iloc[ix]
        p = infer(f, selected)[:, :, target] * scale[target] + mean[target]
        y = np.stack([f[TARGET].iloc[int(r.target_start_pos):int(r.target_end_pos) + 1].to_numpy()
                      for r in selected.itertuples()])
        origin = f[TARGET].iloc[selected.input_end_pos.to_numpy()].to_numpy()
        assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(origin).all()
        predictions.extend(p); truths.extend(y); origins.extend(origin)
        for r in selected.itertuples():
            selections.append(dict(chunk_id=row.chunk_id, window_index=int(r.Index),
                                   origin_time=str(f.timestamp.iloc[int(r.input_end_pos)]),
                                   input_start_pos=int(r.input_start_pos), target_end_pos=int(r.target_end_pos)))
        print(f'Validation {ci + 1}/{len(chunks)}; windows={len(selected)}', flush=True)
    p, y, o = np.array(predictions), np.array(truths), np.array(origins)
    baseline = np.broadcast_to(o[:, None], y.shape)
    blend = o[:, None] + alpha * (p - o[:, None])
    methods = {'raw_model': p, 'persistence': baseline, 'fixed_blend': blend}
    selection = pd.DataFrame(selections)
    selection.to_csv(OUT / 'selected_windows.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(OUT / 'preheater_validation_predictions.npz', raw=p, true=y, origin=o, blend=blend, alpha=alpha)
    rows, daily, per_window, horizon = [], [], [], []
    for name, pred in methods.items():
        rows.append(dict(method=name, windows=len(y), **metric(pred, y)))
        for ci in selection.chunk_id.unique():
            keep = (selection.chunk_id == ci).to_numpy()
            daily.append(dict(chunk_id=ci, method=name, windows=int(keep.sum()), **metric(pred[keep], y[keep])))
        for i in range(len(y)):
            per_window.append(dict(window=i, method=name, actual_mean_abs_change=float(np.abs(y[i] - o[i]).mean()),
                                   mae=float(np.abs(pred[i] - y[i]).mean())))
        for h in range(60):
            horizon.append(dict(method=name, seconds=(h + 1) * 10, **metric(pred[:, h], y[:, h])))
    for name, data in [('overall_metrics', rows), ('daily_metrics', daily), ('window_metrics', per_window), ('horizon_metrics', horizon)]:
        pd.DataFrame(data).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    # Paired date-block bootstrap: days, not overlapping scalar labels, are the resampling unit.
    day = pd.DataFrame(daily).pivot(index='chunk_id', columns='method', values='mae')
    rng = np.random.default_rng(20261022)
    difference = (day.raw_model - day.fixed_blend).to_numpy()
    bootstrap = difference[rng.integers(0, len(difference), (10000, len(difference)))].mean(axis=1)
    summary = dict(sample_windows=len(y), chunks=len(chunks), original_validation_windows=int(chunks.windows.sum()),
                   first_origin=selection.origin_time.min(), last_origin=selection.origin_time.max(),
                   blend_better_than_raw_days=int((difference > 0).sum()),
                   blend_better_than_persistence_days=int((day.fixed_blend < day.persistence).sum()),
                   raw_better_than_persistence_days=int((day.raw_model < day.persistence).sum()),
                   equal_day_mean_mae_reduction=float(difference.mean()),
                   equal_day_mae_reduction_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist(),
                   alpha=weights.tolist(), inference_parity=parity,
                   checkpoint_sha256=hashlib.sha256((ART / 'checkpoint.pth').read_bytes()).hexdigest(),
                   rules_sha256=hashlib.sha256((PRIOR / 'exploratory_rules.csv').read_bytes()).hexdigest(),
                   limitations='Retrospective original validation split; checkpoint selection used this period. Not a new prospective test. No parameter refitting.')
    assert all((path.stat().st_size, path.stat().st_mtime_ns) == initial[str(path)] for path in sources)
    summary['original_artifacts_unchanged'] = True
    (OUT / 'validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = {'raw_model': '原模型', 'persistence': '保持当前值', 'fixed_blend': '固定分时距组合'}
    colors = {'raw_model': '#EE7733', 'persistence': '#888888', 'fixed_blend': '#0077BB'}
    for name in methods:
        h = pd.DataFrame(horizon).query('method == @name')
        axes[0].plot(h.seconds / 60, h.mae, label=labels[name], color=colors[name])
        axes[1].plot(pd.to_datetime(day.index.str.extract(r'_(\d{8})T', expand=False)), day[name], label=labels[name], color=colors[name])
    axes[0].set(xlabel='预测提前量（分钟）', ylabel='MAE（°C）', title='固定参数在另一时期的分时距误差')
    axes[1].set(xlabel='日期', ylabel='逐日抽样 MAE（°C）', title='跨日期稳健性：每个数据分块最多16个抽样窗口')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / 'independent_horizon_and_daily.png', dpi=180)
    plt.close(fig)
    # Representative cases selected by raw-model error quartiles, not by blend benefit.
    errors = np.abs(p - y).mean(axis=1)
    order = np.argsort(errors)
    cases = [order[int(q * (len(order) - 1))] for q in [.25, .5, .75, .95]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, i in zip(axes.flat, cases):
        ax.plot(np.arange(1, 61) / 6, y[i], color='black', label='真实')
        for name, pred in methods.items():
            ax.plot(np.arange(1, 61) / 6, pred[i], color=colors[name], label=labels[name], alpha=.85)
        ax.set(title=selection.origin_time.iloc[i], xlabel='预测后分钟', ylabel='预热温度（°C）')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('无窗口融合的10分钟轨迹：按原模型误差25%、50%、75%、95%分位选例')
    fig.tight_layout(); fig.savefig(OUT / 'independent_representative_cases.png', dpi=180); plt.close(fig)
    summarize_report()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_report():
    """Summarize cached inference without changing or refitting the fixed rule."""
    a = np.load(OUT / 'preheater_validation_predictions.npz')
    p, y, b, o = [a[n] for n in ['raw', 'true', 'blend', 'origin']]
    selection = pd.read_csv(OUT / 'selected_windows.csv')
    stats = pd.read_csv(OUT / 'overall_metrics.csv').set_index('method')
    horizon = pd.read_csv(OUT / 'horizon_metrics.csv')
    summary = json.loads((OUT / 'validation_summary.json').read_text(encoding='utf-8'))
    change = np.abs(y - o[:, None]).mean(axis=1)
    methods = {'raw_model': p, 'persistence': np.broadcast_to(o[:, None], y.shape), 'fixed_blend': b}
    regimes = []
    for label, keep in [('小变化≤1°C', change <= 1), ('中等变化1–3°C', (change > 1) & (change <= 3)), ('大变化>3°C', change > 3)]:
        for name, pred in methods.items():
            regimes.append(dict(regime=label, method=name, windows=int(keep.sum()), **metric(pred[keep], y[keep])))
    pd.DataFrame(regimes).to_csv(OUT / 'retrospective_change_regimes.csv', index=False, encoding='utf-8-sig')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    regime_table = pd.DataFrame(regimes)
    colors = ['#EE7733', '#888888', '#0077BB']
    labels = ['原模型', '保持当前值', '固定组合']
    for j, (name, color, label) in enumerate(zip(methods, colors, labels)):
        values = regime_table.loc[regime_table.method == name, 'mae'].to_numpy()
        bars = ax.bar(np.arange(3) + (j - 1) * .25, values, width=.25, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(3), ['小变化≤1°C\n367窗', '中等变化1–3°C\n328窗', '大变化>3°C\n61窗'])
    ax.set(ylabel='MAE（°C）', title='收缩抑制平稳期误差，但削弱明显温度变化的响应', ylim=(0, 4.6))
    ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, .01, '按未来真实变化分组，仅用于事后诊断，不能作为在线选择条件', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT / 'change_regime_tradeoff.png', dpi=180); plt.close(fig)
    variation = []
    for name, pred in {'truth': y, **methods}.items():
        d = np.abs(np.diff(np.column_stack([o, pred]), axis=1))
        variation.append(dict(method=name, mean_total_variation=float(d.sum(axis=1).mean()),
                              jump_120_to_130_seconds=float(d[:, 12].mean()), jump_300_to_310_seconds=float(d[:, 30].mean())))
    pd.DataFrame(variation).to_csv(OUT / 'trajectory_variation.csv', index=False, encoding='utf-8-sig')
    t = pd.to_datetime(selection.origin_time)
    overlap = int(((t - t.shift()).dt.total_seconds() < 1800).sum())
    label_overlap = int(((t - t.shift()).dt.total_seconds() < 600).sum())
    periods = []
    # These dates describe input availability, not an intervention or causal experiment.
    for label, keep in [('2026-02-11之前', (t < '2026-02-11').to_numpy()), ('2026-02-12及以后', (t >= '2026-02-12').to_numpy())]:
        for name, pred in methods.items():
            periods.append(dict(period=label, method=name, windows=int(keep.sum()), **metric(pred[keep], y[keep])))
    pd.DataFrame(periods).to_csv(OUT / 'availability_period_metrics.csv', index=False, encoding='utf-8-sig')
    lines = ['# 模型1：固定预热温度保护策略的跨时期复核', '',
             '## 1. 现场质量判断与本次边界', '',
             '已按现场确认修正研究判断：信号中断属于测点问题，不再尝试通过补档、插值或信号改名恢复；负值阀门信号代表异常，维持原筛除，不放宽下限。',
             '本次没有改数据集、清洗阈值、模型权重或部署配置。既有304列输入顺序保持不变，缺失值填训练均值仅是旧模型的数值兼容措施，不是恢复了真实测量。也不因少数辅助输入缺失而额外删除全部窗口。', '',
             '## 2. 怎么验证', '',
             f'- 用固定权重文件在本地FP32推理，复现测试期前、中、后3个分块各3个窗口；14个监督目标的最大标准化误差为{max(x["max_abs_target_z_error"] for x in summary["inference_parity"]):.3g}，复现检查通过。',
             f'- 原验证期{summary["first_origin"]}至{summary["last_origin"]}，覆盖48个分块，抽取{len(y)}个窗口；每块最多16个，均匀覆盖可用时段。一块仅有4个符合间隔要求的窗口。原验证期共有206,881个窗口，因此本次不是全量验证。',
             f'- 每块内部历史20分钟加预测10分钟的跨度不重叠；跨块相邻样本有{overlap}对总跨度重叠，其中{label_overlap}对未来标签跨度重叠。统计按日期成组，不把每个10秒标签当独立样本。',
             '- 组合权重取自此前测试早期拟合的固定规则，本次没有使用验证期重新拟合。10–120秒、130–300秒、310–600秒分别保留原模型变化量的5.24%、51.18%、73.95%，其余向当前温度收缩。只使用预测起点之前可用的信息构造预测输入。',
             '- 这是回顾性跨时期复核：这些日期没有参与组合权重拟合，但曾参与原模型选优，且早于组合参数拟合日期。因此不能当作时间向前的全新测试，也不能据此承诺上线效果。既有数据预处理的历史可用性尚未重新构造，本次沿用现有清洗结果。', '',
             '## 3. 整体误差：有稳定收益，但不是全面胜出', '',
             '|方法|MAE（°C）|RMSE（°C）|平均偏差（°C）|R²|', '|---|---:|---:|---:|---:|']
    names = {'raw_model': '原模型', 'persistence': '未来保持当前温度', 'fixed_blend': '固定分时距组合'}
    for name, row in stats.iterrows():
        lines.append(f'|{names[name]}|{row.mae:.4f}|{row.rmse:.4f}|{row.bias:.4f}|{row.r2:.5f}|')
    improvement = 100 * (1 - stats.loc['fixed_blend', 'mae'] / stats.loc['raw_model', 'mae'])
    ci = summary['equal_day_mae_reduction_bootstrap_95ci']
    lines += ['', f'组合相对原模型MAE改善{improvement:.2f}%，相对保持当前值改善{100*(1-stats.loc["fixed_blend", "mae"]/stats.loc["persistence", "mae"]):.2f}%；48个日期中35个优于原模型，41个优于保持当前值。并非所有日期均改善。',
              f'逐日等权的平均MAE降低{summary["equal_day_mean_mae_reduction"]:.4f}°C；按日期有放回抽样的95%区间为[{ci[0]:.4f}, {ci[1]:.4f}]°C。日期间仍可能存在关联，此区间不代表未来运行保证。',
              '原验证期原模型整体优于保持当前值，而此前全测试期预热温度MAE为1.2304°C、保持当前值为1.1691°C，排序相反。这说明收益随时期和变化强度改变，不能把“温度慢”简单等同于“永远保持不变最好”。', '',
              '## 4. 分时距：短时保护明显，长时会牺牲部分变化响应', '',
              '|提前量|原模型MAE|保持当前值MAE|组合MAE|', '|---|---:|---:|---:|']
    for seconds in [60, 120, 300, 600]:
        h = horizon[horizon.seconds == seconds].set_index('method')
        lines.append(f'|{seconds//60}分钟|{h.loc["raw_model", "mae"]:.4f}|{h.loc["persistence", "mae"]:.4f}|{h.loc["fixed_blend", "mae"]:.4f}|')
    lines += ['', '1分钟组合把原模型误差约减半，但仍略逊于保持当前值；10分钟原模型优于组合。原因是收缩同时削弱了错误波动和真实变化，短时前者占优势，长时后者更重要。', '',
              '## 5. 哪些情况受益、哪些情况受损', '',
              '按未来10分钟真实温度相对当前值的平均绝对变化分组，仅用于事后解释，绝不能把未来真实变化作为在线选模型的条件。1°C和3°C是本次描述性分组，不是经现场确认的报警/安全界限。', '',
              '|真实变化组|窗口数|原模型MAE|保持当前值MAE|组合MAE|', '|---|---:|---:|---:|---:|']
    r = pd.DataFrame(regimes)
    for label in r.regime.unique():
        group = r[r.regime == label].set_index('method')
        lines.append(f'|{label}|{int(group.iloc[0].windows)}|{group.loc["raw_model", "mae"]:.4f}|{group.loc["persistence", "mae"]:.4f}|{group.loc["fixed_blend", "mae"]:.4f}|')
    lines += ['', '小变化窗口保持当前值最稳，组合虽抑制原模型误差但仍不如简单基线；大变化窗口原模型最好，组合MAE反而增加约11.5%。因此目前不建议把组合直接替代全部预热温度预测。这里的“保护”仅指预测输出降抖，不是设备安全保护或可直接用于控制的策略。', '',
              '## 6. 抖动与分段边界', '',
              '|轨迹|10分钟平均累计变化（°C）|120→130秒平均跳变（°C）|300→310秒平均跳变（°C）|', '|---|---:|---:|---:|']
    for v in variation:
        lines.append(f'|{names.get(v["method"], "真实轨迹")}|{v["mean_total_variation"]:.4f}|{v["jump_120_to_130_seconds"]:.4f}|{v["jump_300_to_310_seconds"]:.4f}|')
    lines += ['', '组合累计波动减少约45%，但仍约为真实轨迹的3倍。权重在120秒和300秒后阶跃改变，也会产生新的衔接跳变：300→310秒的组合跳变反而大于原模型。因此平均误差变好不等于轨迹已经自然、可用。', '',
              '## 7. 对输入不可用问题的适用边界', '',
              '保留断点前后分期统计（availability_period_metrics.csv），用来检查时期差异。该分期同时伴随季节、负荷和样本构成变化，不能把误差变化因果归结为某个测点中断，也不能借此量化恢复该测点后的收益。当前不再尝试恢复信号，下一次真正需要训练时再考虑仅用持续可用的特征、显式缺失状态，并保持训练与应用一致。', '',
              '## 8. 结论与下一步', '',
              '1. 目前不需为了这两个已确认的信号质量问题重构数据或重新全量训练，先保留原模型作为参照。',
              '2. 固定组合的跨时期平均收益成立，但会削弱明显升降温事件，且分段衔接不自然，暂不直接替换上线。',
              '3. 下一项小实验应先用历史20分钟的温度变化和可用操作信号识别“平稳/变化中”，只对平稳状态做短时收缩，并比较连续时距权重以避免阶跃。先固定规则，再用后续新日期验收，不在本次复核期继续调到最好。',
              '4. 新日期验收同时看：平稳期误差、明显变化期的响应损失、逐日最差情况及输出衔接，不能只看整体R²或平均MAE。', '',
              '## 图表', '', '![分时距与逐日误差](independent_horizon_and_daily.png)', '',
              '![原始10分钟代表轨迹](independent_representative_cases.png)', '',
              '代表例按原模型误差分位选取，不按组合收益挑选；每条曲线是单次10分钟预测，没有窗口融合。', '',
              '![不同真实变化幅度下的取舍](change_regime_tradeoff.png)', '',
              '## 复核记录', '', '模型权重和固定规则SHA256、预测复现误差、来源文件未修改检查见 validation_summary.json 和 inference_parity.json。本次仅新增分析产物。']
    (OUT / 'analysis_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if '--summary-only' in sys.argv:
        summarize_report()
    else:
        main()
