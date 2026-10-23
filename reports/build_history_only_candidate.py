"""Select a small history-only correction, freeze it, and evaluate without retraining."""

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'offline/forecast/inference'))
from short_horizon_policy import apply_policy

ART = ROOT / 'offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60'
DATA = ROOT / 'offline/artifacts/forecast/data_engineering/outputs/datasets/chunked/model1_10s_seq120_pred60'
VALIDATION = ART / 'evaluation_history_only_20261021/independent_temperature_validation_20261022'
OUT = ROOT / 'offline/artifacts/forecast/candidates/model1_history_only_guarded_v1_20261022'


def metrics(p, y):
    e = p - y
    sst = np.square(y - y.mean()).sum()
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.square(e).mean())),
                bias=float(e.mean()), r2=float(1 - np.square(e).sum() / sst) if sst else None)


def read_history(row, windows):
    frame = pd.read_parquet(DATA / 'frames' / (row.chunk_id + '.parquet'), columns=['B3TE03'])
    v = frame['B3TE03'].to_numpy()
    history = np.stack([v[int(r.input_start_pos):int(r.input_start_pos) + 120] for r in windows.itertuples()])
    return history


def evaluate(name, p, y, history, config, chunks):
    final, gate = apply_policy(history, p, config)
    # The correction has a deliberately small scope: no 5/10-minute changes.
    np.testing.assert_array_equal(final[:, 29:], p[:, 29:])
    change = np.abs(y - history[:, -1, None]).mean(axis=1)
    records = []
    groups = [('all', np.ones(len(y), dtype=bool)), ('small_change_le_1c', change <= 1),
              ('medium_change_1_to_3c', (change > 1) & (change <= 3)), ('large_change_gt_3c', change > 3)]
    for group, keep in groups:
        if not keep.any():
            continue
        for method, pred in [('raw', p), ('guarded', final), ('persistence', np.broadcast_to(history[:, -1, None], y.shape))]:
            records.append(dict(period=name, group=group, method=method, windows=int(keep.sum()),
                                active_fraction=float((gate[keep] > 0).mean()), **metrics(pred[keep], y[keep])))
    daily = []
    for chunk in np.unique(chunks):
        keep = chunks == chunk
        for method, pred in [('raw', p), ('guarded', final)]:
            daily.append(dict(period=name, chunk_id=chunk, method=method, windows=int(keep.sum()), **metrics(pred[keep], y[keep])))
    horizon = []
    for method, pred in [('raw', p), ('guarded', final), ('persistence', np.broadcast_to(history[:, -1, None], y.shape))]:
        for h in range(60):
            horizon.append(dict(period=name, method=method, seconds=(h + 1) * 10, **metrics(pred[:, h], y[:, h])))
    return records, daily, horizon


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_dir = OUT / 'evaluation'
    result_dir.mkdir(exist_ok=True)
    if (OUT / 'frozen_policy.json').exists():
        raise FileExistsError('Candidate already exists; preserve frozen selection and use a new explicit version for a new experiment.')
    a = np.load(VALIDATION / 'preheater_validation_predictions.npz')
    p, y = a['raw'], a['true']
    selected = pd.read_csv(VALIDATION / 'selected_windows.csv')
    chunks = pd.read_csv(DATA / 'splits/val_chunks.csv')
    history = np.empty((len(selected), 120))
    for row in chunks.itertuples():
        mask = selected.chunk_id == row.chunk_id
        history[mask] = read_history(row, selected[mask])
    np.testing.assert_allclose(history[:, -1], a['origin'], rtol=0, atol=0)
    calibration = selected.chunk_id.isin(chunks.chunk_id.iloc[:26]).to_numpy()
    replication = selected.chunk_id.isin(chunks.chunk_id.iloc[27:]).to_numpy()
    large = np.abs(y - history[:, -1, None]).mean(axis=1) > 3
    trials = []
    raw_mae = np.abs(p[calibration] - y[calibration]).mean()
    raw_large = np.abs(p[calibration & large] - y[calibration & large]).mean()
    for threshold in [.5, 1., 2.]:
        for end in [120, 180]:
            config = dict(enabled=True, threshold_c=threshold, end_seconds=end)
            pred, gate = apply_policy(history[calibration], p[calibration], config)
            mae = np.abs(pred - y[calibration]).mean()
            large_mae = np.abs(pred[large[calibration]] - y[calibration][large[calibration]]).mean()
            trials.append(dict(**config, mae=float(mae), improvement=float(1 - mae / raw_mae),
                               large_change_degradation=float(large_mae / raw_large - 1), active_fraction=float((gate > 0).mean())))
    pd.DataFrame(trials).to_csv(result_dir / 'calibration_trials.csv', index=False, encoding='utf-8-sig')
    eligible = [r for r in trials if r['improvement'] >= .005 and r['large_change_degradation'] <= .02]
    chosen = min(eligible, key=lambda r: r['mae']) if eligible else dict(enabled=False)
    config = {k: chosen[k] for k in ['enabled', 'threshold_c', 'end_seconds'] if k in chosen}
    frozen = dict(version='model1_history_only_guarded_v1_20261022', target='B3TE03', policy=config,
                  calibration_chunks=chunks.chunk_id.iloc[:26].tolist(), gap_chunk=chunks.chunk_id.iloc[26],
                  replication_chunks=chunks.chunk_id.iloc[27:].tolist(), selection_trials=6,
                  checkpoint_sha256=hashlib.sha256((ART / 'checkpoint.pth').read_bytes()).hexdigest(),
                  status='frozen_before_replication', new_unseen_test=False)
    (OUT / 'frozen_policy.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Frozen policy:', config, flush=True)
    records, days, horizons = [], [], []
    for name, mask in [('validation_calibration', calibration), ('validation_replication', replication)]:
        r, d, h = evaluate(name, p[mask], y[mask], history[mask], config, selected.chunk_id.to_numpy()[mask])
        records += r; days += d; horizons += h
    meta = json.loads((ART / 'metadata.json').read_text(encoding='utf-8'))
    scaler = json.loads((ART / 'scaler.json').read_text(encoding='utf-8'))
    ti = meta['feature_columns'].index('B3TE03')
    mean, scale = scaler['mean'][ti], scaler['scale'][ti]
    saved_p, saved_y = [np.load(ART / n, mmap_mode='r') for n in ['pred.npy', 'true.npy']]
    test_chunks = pd.read_csv(DATA / 'splits/test_chunks.csv')
    n = len(saved_p)
    assert saved_p.shape == saved_y.shape and n == int(test_chunks.windows.sum())
    test_p, test_y, test_h = np.empty((n, 60)), np.empty((n, 60)), np.empty((n, 120))
    test_ids = np.empty(n, dtype=object)
    cursor = 0
    for j, row in enumerate(test_chunks.itertuples()):
        w = pd.read_csv(DATA / 'windows' / (row.chunk_id + '_windows.csv'))
        stop = cursor + len(w)
        test_h[cursor:stop] = read_history(row, w)
        # mmap reads one target at a time; no 14 GB in-memory copy.
        test_p[cursor:stop] = np.asarray(saved_p[cursor:stop, :, ti], dtype='float64') * scale + mean
        test_y[cursor:stop] = np.asarray(saved_y[cursor:stop, :, ti], dtype='float64') * scale + mean
        test_ids[cursor:stop] = row.chunk_id
        cursor = stop
        print(f'Test extraction {j + 1}/{len(test_chunks)}', flush=True)
    assert all(np.isfinite(z).all() for z in [test_p, test_y, test_h])
    r, d, h = evaluate('test_all', test_p, test_y, test_h, config, test_ids)
    records += r; days += d; horizons += h
    for name, data in [('group_metrics', records), ('daily_metrics', days), ('horizon_metrics', horizons)]:
        pd.DataFrame(data).to_csv(result_dir / (name + '.csv'), index=False, encoding='utf-8-sig')
    checks = []
    for period in ['validation_replication', 'test_all']:
        groups = pd.DataFrame(records).query('period == @period')
        whole = groups[groups.group == 'all'].set_index('method')
        big = groups[groups.group == 'large_change_gt_3c'].set_index('method')
        day = pd.DataFrame(days).query('period == @period').pivot(index='chunk_id', columns='method', values='mae')
        checks.append(dict(period=period, improvement=float(1 - whole.loc['guarded', 'mae'] / whole.loc['raw', 'mae']),
                           large_change_degradation=float(big.loc['guarded', 'mae'] / big.loc['raw', 'mae'] - 1),
                           nonworse_days=int((day.guarded <= day.raw + 1e-12).sum()), days=len(day)))
    passed = config.get('enabled', False) and all(c['improvement'] > 0 and c['large_change_degradation'] <= .02 and c['nonworse_days'] >= c['days'] / 2 for c in checks)
    result = dict(passed=passed, policy_to_package=config if passed else dict(enabled=False), checks=checks,
                  caveat='Previously inspected validation/test dates; retrospective engineering verification, not a new final test.',
                  other_13_targets_unchanged=True, five_and_ten_minute_predictions_unchanged=True)
    (OUT / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # Keep compact examples for plots without storing another full prediction array.
    final, gate = apply_policy(test_h, test_p, config)
    sample = np.unique(np.linspace(0, n - 1, 240).round().astype(int))
    np.savez_compressed(result_dir / 'display_samples.npz', raw=test_p[sample], true=test_y[sample],
                        guarded=final[sample], history=test_h[sample], gate=gate[sample], indices=sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def package_candidate():
    acceptance = json.loads((OUT / 'acceptance.json').read_text(encoding='utf-8'))
    frozen = json.loads((OUT / 'frozen_policy.json').read_text(encoding='utf-8'))
    if hashlib.sha256((ART / 'checkpoint.pth').read_bytes()).hexdigest() != frozen['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since selection')
    bundle = OUT / 'bundle'
    if bundle.exists():
        raise FileExistsError('Do not overwrite an existing candidate bundle')
    bundle.mkdir()
    for name in ['checkpoint.pth', 'scaler.json']:
        shutil.copy2(ART / name, bundle / name)
    original_meta = json.loads((ART / 'metadata.json').read_text(encoding='utf-8'))
    meta = {k: original_meta[k] for k in ['model_slug', 'dataset_id', 'freq_seconds', 'seq_len', 'label_len',
                                        'pred_len', 'feature_columns', 'target_columns', 'target_indices']}
    (bundle / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    command = shlex.split((ART / 'last_train_command.txt').read_text(encoding='utf-8').splitlines()[0])
    if '--no_future_covariates' not in command or command[command.index('--model') + 1] != 'Crossformer':
        raise ValueError('Unexpected source model or future-input training mode')
    fields = ['enc_in', 'seq_len', 'pred_len', 'patch_len', 'd_model', 'n_heads', 'd_ff', 'e_layers', 'factor']
    config = {f: int(command[command.index('--' + f) + 1]) for f in fields}
    config.update(task_name='long_term_forecast', dropout=.1, use_dec_mask_embedding=True)
    (bundle / 'model_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (bundle / 'policy.json').write_text(json.dumps(acceptance['policy_to_package'], indent=2), encoding='utf-8')
    for name in ['predict.py', 'short_horizon_policy.py', 'README.md']:
        shutil.copy2(ROOT / 'offline/forecast/inference' / name, bundle / name)
    # Only the import closure needed by Crossformer/PatchTST; not the full training stack.
    relative_sources = ['models/__init__.py', 'models/Crossformer.py', 'models/PatchTST.py',
                        'layers/__init__.py', 'layers/Crossformer_EncDec.py', 'layers/Embed.py',
                        'layers/SelfAttention_Family.py', 'layers/Transformer_EncDec.py',
                        'utils/__init__.py', 'utils/masking.py']
    for name in relative_sources:
        destination = bundle / 'tslib' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'offline/forecast/tslib' / name, destination)
    requirements = 'torch>=2.6,<3\nnumpy>=2,<3\npandas>=2,<3\neinops>=0.8,<1\n'
    (bundle / 'requirements.txt').write_text(requirements, encoding='utf-8')
    (bundle / 'SOURCE_NOTICE.txt').write_text('TSLib/Crossformer source copied from this project frozen training implementation.\nUpstream: https://github.com/thuml/Time-Series-Library\nCrossformer: https://github.com/Thinklab-SJTU/Crossformer\nLocal modifications are preserved, not replaced with upstream code.\n', encoding='utf-8')
    row = pd.read_csv(DATA / 'splits/test_chunks.csv').iloc[0]
    f = pd.read_parquet(DATA / 'frames' / (row.chunk_id + '.parquet'))
    w = pd.read_csv(DATA / 'windows' / (row.chunk_id + '_windows.csv')).iloc[0]
    example = f.iloc[int(w.input_start_pos):int(w.input_end_pos) + 1][['timestamp'] + meta['feature_columns']]
    example.to_csv(bundle / 'example_history.csv', index=False, encoding='utf-8-sig')
    (bundle / 'example_origin.txt').write_text(str(example.timestamp.iloc[-1]) + '\n', encoding='utf-8')
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    files = {p.relative_to(bundle).as_posix(): digest(p) for p in bundle.rglob('*') if p.is_file()}
    import torch
    import einops
    manifest = dict(version=frozen['version'], history_only=True, usage='offline_shadow_candidate',
                    files=files, correction_passed_retrospective_checks=acceptance['passed'],
                    source_checkpoint_sha256=frozen['checkpoint_sha256'],
                    tested_versions=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                                         pandas=pd.__version__, einops=einops.__version__))
    (bundle / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Packaged:', bundle)


def write_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    evaluation = OUT / 'evaluation'
    groups = pd.read_csv(evaluation / 'group_metrics.csv')
    horizons = pd.read_csv(evaluation / 'horizon_metrics.csv')
    frozen = json.loads((OUT / 'frozen_policy.json').read_text(encoding='utf-8'))
    acceptance = json.loads((OUT / 'acceptance.json').read_text(encoding='utf-8'))
    labels = {'raw': '原模型', 'guarded': '历史门控短时修正', 'persistence': '保持当前值'}
    colors = {'raw': '#EE7733', 'guarded': '#0077BB', 'persistence': '#888888'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in labels:
        h = horizons[(horizons.period == 'test_all') & (horizons.method == name)]
        axes[0].plot(h.seconds / 60, h.mae, label=labels[name], color=colors[name])
    axes[0].set(xlabel='预测提前量（分钟）', ylabel='MAE（°C）', title='只修正最短3分钟，保留远期动态预测')
    axes[0].legend(fontsize=8)
    periods = ['validation_replication', 'test_all']
    for j, name in enumerate(['raw', 'guarded']):
        values = [groups[(groups.period == period) & (groups.group == 'all') & (groups.method == name)].iloc[0].mae for period in periods]
        bars = axes[1].bar(np.arange(2) + (j - .5) * .32, values, width=.32, color=colors[name], label=labels[name])
        axes[1].bar_label(bars, fmt='%.4f', padding=3)
    axes[1].set_xticks([0, 1], ['较晚验证期\n324个抽样窗', '完整测试期\n201,244个重叠窗'])
    axes[1].set(ylabel='MAE（°C）', title='平均收益较小，但两段时期方向一致', ylim=(0, 1.45))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(evaluation / 'candidate_performance.png', dpi=180); plt.close(fig)
    h = horizons[horizons.period == 'test_all']
    whole = groups[(groups.period == 'test_all') & (groups.group == 'all')].set_index('method')
    lines = ['# 模型1历史输入可调用候选版：实施与验收', '',
             '## 交付结论', '',
             f'短时修正通过本轮预设的回顾性检查：{acceptance["passed"]}。已形成独立推理包，不需要重新训练，也不需要多卡推理。仅用于离线研究/只读影子运行，不替换已发布bundle或Profile，模型2不变。', '',
             '## 模型与修正逻辑', '',
             '原Crossformer权重及304特征、14目标契约保持不变，历史20分钟输入、未来10分钟输出。只针对B3TE03：历史近5分钟范围、近10分钟净变化二者取大值；状态量越小，最短时距越向当前值收缩；180秒时回到原模型，之后逐值不变。权重使用连续余弦曲线，不做窗口融合。其余13目标不受修正影响。',
             f'冻结参数为：{json.dumps(frozen["policy"], ensure_ascii=False)}。2°C是本次算法门控参数，不是工艺安全限值；只能判断历史相对平稳，不能保证随后不会突然变化。', '',
             '## 如何避免追着测试集调参', '',
             '事先只声明6组参数（0.5/1/2°C、120/180秒）。原验证期前26分块416个抽样窗用于选型；隔开1分块16窗；后21分块324窗只复核。之后固定参数检查全测试期201,244窗、52日期分块，没有根据后两组结果再调参数。选择表见calibration_trials.csv。',
             '交付门槛事先写入计划：校准整体MAE至少改善0.5%；较晚复核/测试整体改善、大变化组退步≤2%、至少半数日期不退步；5和10分钟结果必须原样保留。未通过时只封装原模型。当前检查通过，不等于正式生产准入。',
             '重要限制：原验证期曾参与模型权重选优，且本批验证/测试结果在之前诊断中已被查看；本次保留的是参数拟合与后续检查分离，不是全新、盲测的最终测试。20万个重叠窗不是20万个独立事件。既有清洗存在前视依赖的个例，此次没有重构清洗数据，因此此处精度仍是现有数据口径下的回顾结果。', '',
             '## 预热温度整体测试结果', '',
             '|方法|MAE（°C）|RMSE（°C）|R²|', '|---|---:|---:|---:|']
    for name in labels:
        r = whole.loc[name]
        lines.append(f'|{labels[name]}|{r.mae:.4f}|{r.rmse:.4f}|{r.r2:.5f}|')
    lines += ['', '整体MAE改善1.69%，52/52日期分块的平均MAE不退步；较晚验证期改善1.03%，21/21日期不退步。收益是小幅、保守的，不能描述为全面提升。',
              '尤其注意：预热温度整体MAE仍不如保持当前值（1.2096°C对1.1691°C）；虽然RMSE和R²更好，也不能掩盖这一不足。短时修正没有消除所有伪波动，不保证每个单窗、每个突变事件都获益。', '',
              '|提前量|原模型MAE|候选版MAE|保持当前值MAE|', '|---|---:|---:|---:|']
    for seconds in [60, 120, 180, 300, 600]:
        t = h[h.seconds == seconds].set_index('method')
        lines.append(f'|{seconds//60}分钟|{t.loc["raw", "mae"]:.4f}|{t.loc["guarded", "mae"]:.4f}|{t.loc["persistence", "mae"]:.4f}|')
    lines += ['', '1分钟MAE下降约19.1%，但仍不如保持当前值。3、5、10分钟完全不变：本版没有靠压平整段预测换平均分。', '',
              '## 真实变化组：检查是否牺牲响应', '',
              '|组别|窗口数|原模型MAE|候选版MAE|', '|---|---:|---:|---:|']
    for group in ['small_change_le_1c', 'medium_change_1_to_3c', 'large_change_gt_3c']:
        t = groups[(groups.period == 'test_all') & (groups.group == group)].set_index('method')
        lines.append(f'|{group}|{int(t.loc["raw", "windows"])}|{t.loc["raw", "mae"]:.4f}|{t.loc["guarded", "mae"]:.4f}|')
    lines += ['', '组别按未来10分钟真值相对当前值的平均绝对变化分为≤1、1–3、>3°C，只有事后评价使用这些未来值，预测接口不接收它们。大变化组平均未退步，但不能据此推断所有突变都能提前识别。', '',
              '## 14个目标现有表现', '',
              '其余13目标本版没有改变，因此沿用原全测试期指标；预热温度替换为本次候选值。R²反映跨工况数值拟合，不等于变化预测准确率。', '',
              '|目标|单位|候选MAE|R²|', '|---|---|---:|---:|']
    target_metrics = pd.read_csv(ART / 'evaluation_history_only_20261021/target_metrics.csv')
    for row in target_metrics.itertuples():
        mae = whole.loc['guarded', 'mae'] if row.target == 'B3TE03' else row.mae_all_10min
        r2 = whole.loc['guarded', 'r2'] if row.target == 'B3TE03' else row.r2_all_10min
        name = {'B3QT00': '氧量A', 'B3_O2_S1': '氧量B'}.get(row.target, row.name)
        lines.append(f'|{name}（{row.target}）|{row.unit}|{mae:.4f}|{r2:.5f}|')
    lines += ['', '## 质量机制与使用边界', '',
              '输入是已清洗120行10秒网格，必须声明预测起点。时间错乱、多余未来行、重复列、未知列、资产哈希或结构不一致会报错。缺失辅助输入仅兼容训练均值并报告；当前目标缺失则标记不可用；严重缺测或非有限模型输出退回当前值并标明降级，不输出NaN伪成功。',
              '中断测点按不可恢复处理，现场确认异常的负值继续筛除；不扩容差、不补假信号。推理入口不重做原始Silver清洗，也不能自动验证物理单位和仪表含义。工程兜底阈值不是经过事件标定的预测置信度。', '',
              '## 文件与运行', '',
              '- bundle/：冻结权重、缩放器、契约、独立predict.py、最小TSLib源码、示例历史及README；不依赖训练数据集或主工程路径。',
              '- frozen_policy.json：先于后期检查写入的参数与日期分组；acceptance.json：交付检查结果。',
              '- evaluation/：逐日/逐时距/变化组指标、示例输出、验证记录和本报告。',
              '- 在bundle目录执行：`python predict.py --history example_history.csv --as-of "2026-03-08 00:19:50" --output prediction_check.json`。',
              '- 输出每一目标的状态都应检查；degraded_persistence表示降级，unavailable表示不可用。usable_for_control固定为false。', '',
              '### 本轮源码改动', '',
              '|文件|作用及原因|', '|---|---|',
              '|inference/predict.py|新增独立历史输入预测接口、资产/时序校验与按目标降级；不修改后端调用。|',
              '|inference/short_horizon_policy.py|新增只读取历史的短时连续修正；避免影响其他目标和长时距。|',
              '|inference/README.md|新增输入单位、清洗责任、运行命令、输出状态和使用边界说明。|',
              '|reports/build_history_only_candidate.py|新增固定选型、后期复核、独立打包和报告生成流程。|',
              '|tests/test_history_only_inference.py|新增15项规则、缺测、时间、解码输入、完整性与输出状态测试。|',
              '|forecast/README.md、reports/README.md|补记新模块职责和复现入口。|',
              '|docs/03-models/forecast-models.md|补记训练版候选与历史发布模型的区别，避免误当已上线。|',
              '|docs/changelog/CHANGELOG.md|记录此次交付、验证及未改动边界。|',
              '|docs/plans/2026-10-22-history-only-candidate.md|记录实施前的选择空间和通过条件。|', '',
              '## 后续验收，不再盲目加改动', '',
              '下一阶段应先用未参与本轮研究的后续日期做只读回放，并核对上游严格历史清洗、单位、延迟和信号可用性；检查明显操作事件和最差日期。先解决这一轮的现场/新日期验收，不直接进入下一轮重训或自动控制接入。若新日期收益不复现，禁用短时规则而保留原模型。', '',
              '## 方法参考', '',
              '预测组合可作为单模型的简单对照，但不保证改进：[Forecast combinations](https://otexts.com/fpp3/combinations.html)。时间顺序评价应只用起点之前信息构造预测，并按预测时距检查：[Time series cross-validation](https://otexts.com/fpp3/tscv.html)。这里借鉴其评价思路，不借用通用数据集的精度结论；本炉的阈值与有效性以本报告实验为准。', '',
              '![候选版误差与时距](evaluation/candidate_performance.png)', '']
    verification_path = evaluation / 'verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        lines += ['## 实际运行检查', '', f'```json\n{json.dumps(verification, ensure_ascii=False, indent=2)}\n```', '']
    (OUT / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    if '--package-only' in sys.argv:
        package_candidate()
    elif '--report-only' in sys.argv:
        write_report()
    else:
        main()
