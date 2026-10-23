"""Report successful/failed evidence without promoting a failed oxygen correction."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import improve_oxygen_history_only as experiment
from oxygen_correction import correct_oxygen


def preview_rolling_bias(a):
    """Early test-period feasibility only; all residual labels have matured by origin."""
    chunks = pd.read_csv(experiment.DATA / 'splits/test_chunks.csv').iloc[:26]
    count = int(chunks.windows.sum())
    p, h, y = a['test_pred'][:count], a['test_history'][:count], a['test_true'][:count]
    result, cursor = p.copy(), 0
    alpha = 1. - np.exp(-10. / 180.)
    for row in chunks.itertuples():
        w = pd.read_csv(experiment.DATA / 'windows' / (row.chunk_id + '_windows.csv'))
        times = pd.to_datetime(w.input_end_time).to_numpy()
        bounds = np.r_[0, np.flatnonzero(np.diff(times) != np.timedelta64(10, 's')) + 1, len(times)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            raw, current = p[cursor + lo:cursor + hi], h[cursor + lo:cursor + hi, -1]
            for step in range(1, min(60, len(raw) - 6) + 1):
                residual = current[step:] - raw[:-step, step - 1]
                # EMA of past forecasts verified by the currently observed value.
                bias = pd.DataFrame(np.vstack([np.zeros((1, 2)), residual])).ewm(alpha=alpha, adjust=False).mean().to_numpy()[1:]
                result[cursor + lo + step + 5:cursor + hi, step - 1] += .5 * np.clip(bias[5:], -.5 * a['target_scale'], .5 * a['target_scale'])
        cursor += len(w)
    improvement = [float(1 - np.abs(result[:, :, j] - y[:, :, j]).mean() / np.abs(p[:, :, j] - y[:, :, j]).mean()) for j in range(2)]
    return dict(strength=.5, decay_seconds=180, warmup_residuals=6, cap_training_std_fraction=.5,
                windows=count, improvement_a=improvement[0], improvement_b=improvement[1],
                status='exploratory early-period only; not selected or tested on later dates')


def main():
    out = experiment.OUT
    state = out / 'state_calibration'
    a = np.load(out / 'oxygen_evaluation_cache.npz')
    p, y, history = [a[k] for k in ['test_pred', 'test_true', 'test_history']]
    frozen = json.loads((state / 'frozen_policy.json').read_text(encoding='utf-8'))
    acceptance = json.loads((state / 'acceptance.json').read_text(encoding='utf-8'))
    records = pd.read_csv(state / 'group_metrics.csv')
    horizons = pd.read_csv(state / 'horizon_metrics.csv')
    prior = experiment.ART / 'evaluation_history_only_20261021/diagnostics_20261021'
    context = pd.read_parquet(prior / 'window_diagnostics.parquet', columns=['window_id', 'target_end_time', 'current_steam', 'past_control_range_score', 'mae_B3QT00'])
    np.testing.assert_array_equal(context.window_id.to_numpy(), np.arange(len(p)))
    np.testing.assert_allclose(context.mae_B3QT00, np.abs(p[:, :, 0] - y[:, :, 0]).mean(axis=1), atol=1e-5)
    prior_meta = json.loads((prior / 'diagnostic_metadata.json').read_text(encoding='utf-8'))
    load_cuts = prior_meta['load_train_tertile_cuts']
    action_cuts = prior_meta['past_action_calibration_cuts']
    corrected = np.empty_like(p)
    diagnostics, lags, groups, cases, transitions = [], [], [], [], []
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, code in enumerate(experiment.CODES):
        corrected[:, :, j], _ = correct_oxygen(history[:, :, j], p[:, :, j], frozen[code])
        for method, color, label in [('raw', '#EE7733', '原模型'), ('corrected', '#888888', '未通过验收的状态校准'), ('persistence', '#0077BB', '保持当前值')]:
            h = horizons[(horizons.period == 'test_all') & (horizons.target == code) & (horizons.method == method)]
            axes[j].plot(h.seconds / 60, h.mae, color=color, label=label)
        axes[j].set(title=code, xlabel='预测提前量（分钟）', ylabel='MAE（百分点）')
        axes[j].legend(fontsize=8); axes[j].grid(alpha=.2)
        error = p[:, :, j] - y[:, :, j]
        first_offset = p[:, 0, j] - history[:, -1, j]
        diagnostics.append(dict(target=code, window_offset_mse_share=float(60 * np.square(error.mean(axis=1)).sum() / np.square(error).sum()),
                                first_prediction_vs_current_abs_p99=float(np.quantile(np.abs(first_offset), .99)),
                                corrected_min=float(corrected[:, :, j].min()), corrected_max=float(corrected[:, :, j].max())))
        # 60-second increments suppress patch-scale noise; displacement is diagnostic only.
        dp, dy = p[:, 6:, j] - p[:, :-6, j], y[:, 6:, j] - y[:, :-6, j]
        for lag in range(-6, 7):
            pp, yy = (dp[:, lag:], dy[:, :-lag]) if lag > 0 else ((dp[:, :lag], dy[:, -lag:]) if lag < 0 else (dp, dy))
            pp, yy = pp.ravel(), yy.ravel()
            pp, yy = pp - pp.mean(), yy - yy.mean()
            corr = float(np.dot(pp, yy) / np.sqrt(np.dot(pp, pp) * np.dot(yy, yy)))
            lags.append(dict(target=code, predicted_delay_seconds=lag * 10, change_correlation=corr))
        load_group = np.digitize(context.current_steam, load_cuts)
        action_group = np.digitize(context.past_control_range_score, action_cuts)
        oxygen_z = (history[:, -1, j] - frozen[code]['training_mean']) / frozen[code]['training_scale']
        past_delta = history[:, -1, j] - history[:, -31, j]
        future_delta = y[:, -1, j] - history[:, -1, j]
        threshold = .1 * a['target_scale'][j]
        eligible = np.abs(future_delta) >= threshold
        past_active = np.abs(past_delta) >= threshold
        transition_masks = {'continuation': eligible & past_active & (np.sign(past_delta) == np.sign(future_delta)),
                            'reversal': eligible & past_active & (np.sign(past_delta) != np.sign(future_delta)),
                            'emerging': eligible & ~past_active}
        for label, keep in transition_masks.items():
            transitions.append(dict(target=code, transition=label, windows=int(keep.sum()),
                                    **experiment.metrics(p[keep, :, j], y[keep, :, j], history[keep, -1, j], threshold)))
        for label, membership in [('load', load_group), ('past_controls', action_group), ('oxygen_level', np.digitize(oxygen_z, [-1., 1.]))]:
            for category in np.unique(membership):
                keep = membership == category
                for method, pred in [('raw', p[:, :, j]), ('unaccepted_state_calibration', corrected[:, :, j])]:
                    groups.append(dict(target=code, context=label, group=int(category), method=method, windows=int(keep.sum()),
                                       **experiment.metrics(pred[keep], y[keep, :, j], history[keep, -1, j], .1 * a['target_scale'][j])))
        # Cases are selected by raw error quantiles within rising/falling events, not benefit.
        delta = y[:, -1, j] - history[:, -1, j]
        figure, panels = plt.subplots(2, 2, figsize=(12, 7))
        for axis, (direction, quantile) in zip(panels.flat, [('rise', .5), ('fall', .5), ('rise', .95), ('fall', .95)]):
            eligible = np.flatnonzero(delta >= .1 * a['target_scale'][j] if direction == 'rise' else delta <= -.1 * a['target_scale'][j])
            mae = np.abs(p[eligible, :, j] - y[eligible, :, j]).mean(axis=1)
            idx = eligible[np.argsort(mae)[int(quantile * (len(eligible) - 1))]]
            current = history[idx, -1, j]
            xx = np.arange(61) / 6
            axis.plot(xx, np.r_[current, y[idx, :, j]], color='#0077BB', label='真实')
            axis.plot(xx, np.r_[current, p[idx, :, j]], color='#EE7733', label='原预测')
            axis.plot(xx, np.r_[current, corrected[idx, :, j]], color='#888888', ls='--', label='未通过方案')
            origin = pd.Timestamp(context.target_end_time.iloc[idx]) - pd.Timedelta(seconds=600)
            axis.set(title=f'{direction} / 原误差{quantile:.0%}分位 / {origin}', xlabel='预测后分钟', ylabel='氧量（%）')
            axis.grid(alpha=.2); axis.legend(fontsize=8)
            cases.append(dict(target=code, window_id=int(idx), direction=direction, raw_error_quantile=quantile, origin=str(origin)))
        figure.suptitle(f'{code}：单次10分钟预测，无窗口融合，不按修正收益挑选案例')
        figure.tight_layout(); figure.savefig(out / f'cases_{code}.png', dpi=170); plt.close(figure)
    fig.suptitle('校准未明显解决氧量误差：原模型仍为推荐输出')
    fig.tight_layout(); fig.savefig(out / 'oxygen_horizon_errors.png', dpi=180); plt.close(fig)
    for name, rows in [('error_components', diagnostics), ('lag_diagnostics', lags), ('context_metrics', groups), ('case_selections', cases), ('trend_transition_metrics', transitions)]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False, encoding='utf-8-sig')
    source = json.loads((out / 'source_provenance.json').read_text(encoding='utf-8'))
    assert source['artifacts'] == experiment.source_signature()
    selected = json.loads((state / 'acceptance.json').read_text(encoding='utf-8'))['oxygen_policy']
    assert all(c['method'] == 'none' for c in selected.values())
    summary = dict(recommend_replace=False, applied_oxygen_policy=selected,
                   original_weights_and_arrays_unchanged=True, v1_bundle_unchanged=True,
                   test_future_change_correlation=float(np.corrcoef((y[:, -1] - history[:, -1]).T)[0, 1]),
                   raw_error_correlation=float(np.corrcoef((p[:, -1] - y[:, -1]).T)[0, 1]),
                   new_model_bundle_created=False, no_retraining=True,
                   optional_early_test_rolling_bias_preview=preview_rolling_bias(a))
    (out / 'diagnostic_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 两路氧量：不重训校准的诊断与否决结果', '',
             '## 结论', '', '本轮没有找到通过完整检查的氧量修正，不生成或推荐新的氧量模型包。继续使用原历史输入网络及此前仅针对一次风预热温度的v1候选；不能把本实验目录理解为已完成的v2模型。',
             'B3TE03是一次风预热器出口温度，不是炉膛温度。本研究仅针对B3QT00（氧量A）、B3_O2_S1（氧量B），未优化其他12个目标，也未更换模型权重。', '',
             '## 1. 做过哪些检查', '',
             '- 用原权重补跑原验证期756个抽样窗，并在测试期前、中、后各3窗复现两路氧量保存预测；全测试期201,244窗用于固定方案后期评价。',
             '- 前26验证分块416窗确定参数，隔1分块16窗，后21分块324窗复核；原验证期曾用于模型选优，这些测试数据此前也已被观察，因此不是新的盲测。',
             '- 沿用现有清洗数据；此前发现过局部前视清洗依赖，本轮未重构，不能把这些结果当作严格在线清洗链已经验收。',
             '- 第一阶段比较原预测、常数中位数偏差、4组当前氧量起点补偿，共6项。均未达到校准期MAE改善至少1%的要求。',
             '- 第二阶段只增加一个明确假设：误差可能与当前氧量和预测变化量有关。每路拟合3个线性标定参数，比较3个预设约束强度，参数在后期检查前冻结，不重训Crossformer。',
             '- 后期同时要求MAE改善、RMSE退步≤2%、变化方向准确率下降≤1个百分点，终点上升/下降及前一分钟快速升降场景MAE退步≤2%，至少半数日期平均不退步。',
             '- 补充早期测试期连续回放仅做可行性预览：已兑现残差的滚动半强度纠偏改善约0.22%/0.33%，未继续在较晚日期选型或封装。该预览不能作为可用模型成果。', '',
             '## 2. 原模型到底差在哪里', '',
             '|目标|整体MAE（百分点）|R²|10分钟方向准确率|10分钟变化幅度系数|', '|---|---:|---:|---:|---:|']
    for code in experiment.CODES:
        row = records[(records.target == code) & (records.period == 'test_all') & (records.group == 'all') & (records.method == 'raw')].iloc[0]
        lines.append(f'|{code}|{row.mae:.4f}|{row.r2:.4f}|{row.direction_accuracy:.2%}|{row.endpoint_change_gain:.3f}|')
    lines += ['', '方向仅在真实10分钟净变化超过训练标准差0.1倍的窗口计算，不是全部窗口的分类准确率。变化幅度系数是“预测净变化对真实净变化”的过原点线性系数，约0.4说明总体响应偏弱，但包含方向错误和不可预测变化，不能直接把曲线乘2.5来补偿。',
              '误差中约61%的平方量可分解为事后整窗偏移，但这种偏移需要未来真值才能计算，不等于当时已知的固定偏差。实际首点相对当前观测本就较准，简单锚定起点并不能知道后续10分钟向哪里变化。', '',
              '## 3. 固定偏差并不跨时期稳定', '', '|时期|氧量A平均偏差|氧量B平均偏差|', '|---|---:|---:|']
    for period in ['validation_calibration', 'validation_replication', 'test_all']:
        biases = [records[(records.target == c) & (records.period == period) & (records.group == 'all') & (records.method == 'raw')].iloc[0].bias for c in experiment.CODES]
        lines.append(f'|{period}|{biases[0]:+.4f}|{biases[1]:+.4f}|')
    lines += ['', '偏差定义为预测减真实。校准期与测试期的方向发生变化，因此用一个固定数值整体抬高或降低，不能保证在后续时期有效。负荷和控制变化分组见context_metrics.csv；这些关联不能证明某个控制动作造成误差，也不应使用未来动作选择当前修正。', '',
              '## 4. 线性状态校准：平均分略好，但回升响应受损', '',
              '|目标/时期|原MAE|实验MAE|相对改善|方向准确率变化|', '|---|---:|---:|---:|---:|']
    for check in acceptance['checks']:
        g = records[(records.target == check['target']) & (records.period == check['period']) & (records.group == 'all')].set_index('method')
        lines.append(f'|{check["target"]}/{check["period"]}|{g.loc["raw", "mae"]:.4f}|{g.loc["corrected", "mae"]:.4f}|{check["improvement"]:.2%}|{check["direction_accuracy_change"]*100:+.2f}个百分点|')
    lines += ['', '较晚验证期回升场景MAE增加约3.05%（A）和3.24%（B），前一分钟快速回升增加约2.48%和3.39%，超过预设上限；方向准确率也下降约1.02/1.36个百分点。全测试期平均MAE仅改善0.63%/0.55%，不足以抵消这类事件损伤，故两路均不启用。',
              '这是一个有用的否决结论：氧量的变化本身重要，不应为减小平均误差继续把回升、下降压弱。此次实际只尝试简单偏移和一次小参数状态校准，没有反复改复杂模型。', '',
              '## 5. 时滞诊断的解释边界', '',
              '在本次±60秒位移范围内，两路变化相关性的最大值均出现在0秒位移，约为0.243/0.246。因此本项检查不支持把预测曲线统一向前移动作为解法；也没有据此认定现场仪表不存在时延。',
              'lag_diagnostics.csv在预测时距轴上比较60秒变化量，尝试±60秒位移；正数表示预测变化在更晚时距才对齐。它只用于定位形状差异，不等于现场烟气传输时延，也不是可部署的时间平移收益。不能据此把模型曲线整体向前搬，尤其不能用未来真值选择每个窗口的最佳位移。', '',
              '## 6. 两路信号及下一步', '',
              f'全测试期两路真实10分钟净变化相关系数为{summary["test_future_change_correlation"]:.6f}，终点预测误差相关系数为{summary["raw_error_correlation"]:.6f}。它们可能共享强过程关系、测量来源或换算关系，但现有数据不能确定；不互相平均替代，也不当作两份独立验证。',
              '当前更值得解决的是稳定的信息来源，而非继续搬动曲线：核对两路测点定义、采样及烟气分析链时序；为预测起点记录真实可用的配风/给料指令、自动/手动模式和设备状态。已确认损坏的信号仍筛除，不靠插值伪造恢复。',
              '下一阶段先解决本轮未通过的事件响应与跨时期偏差问题，不直接进入全量重训。若现场已有未纳入的模式/操作时间记录，应先补作误差分组；若仍无法分辨这些状态，才讨论针对氧量的受控小规模微调或增加有效历史信息。已有后处理证据不足以承诺不重训就能显著提高精度。', '',
              '## 7. 进一步规律：更会预测回摆，不擅长持续单向变化', '',
              '用过去5分钟净变化与随后10分钟净变化，按同一训练标准差门槛分类。两段同向为延续，反向为反转，过去不明显而未来明显为新出现；这是事后诊断，不能作为在线选择条件。', '',
              '|目标|场景|窗口数|原模型MAE|方向准确率|', '|---|---|---:|---:|---:|']
    for row in transitions:
        label = {'continuation': '同向延续', 'reversal': '反转/回摆', 'emerging': '新出现变化'}[row['transition']]
        lines.append(f'|{row["target"]}|{label}|{row["windows"]}|{row["mae"]:.4f}|{row["direction_accuracy"]:.2%}|')
    lines += ['', '原模型对反转/回摆组的方向判断约81%，对持续同向组仅约55%。难点不只是“预测晚了”，还在于区分这次波动将被回调，还是会继续发展。该现象与闭环过程经常回调的直觉相容，但现有观察性数据不能证明控制反馈就是原因。需要模式/操作状态信息来验证，而不是用未来真实场景给模型开后门。',
              '难例：2026-04-13 05:53:10起点的一次10分钟预测，氧量A实际从约4.0%升到5.4%，模型却降到约3.6%。它属于原误差95%分位的困难窗口，不代表全部样本；但说明仅修偏差很难修复方向相反的轨迹。', '',
              '## 图与复现', '',
              '![逐时距误差，灰色方案未启用](oxygen_horizon_errors.png)', '',
              '![氧量A典型及困难窗口](cases_B3QT00.png)', '',
              '![氧量B典型及困难窗口](cases_B3_O2_S1.png)', '',
              '案例按原误差分位在升/降组选择，不按实验收益挑图；单个10分钟窗口，没有融合。', '',
              '源码：improve_oxygen_history_only.py负责数据和第一阶段；calibrate_oxygen_state.py负责第二阶段固定标定与复核；oxygen_correction.py及test_oxygen_correction.py记录实验规则与边界测试；report_oxygen_diagnostics.py生成本报告。预测入口predict.py、v1包、权重、数据集和模型2未改。',
              '所有中间结果保留在本实验目录，包括失败参数与冻结记录；oxygen_evaluation_cache.npz仅用于复算，不是新数据集或新模型。', '',
              '方法参考：评价须与简单基线比较，并按时间顺序拆分参数选择与复核，见[Forecast accuracy](https://otexts.com/fpp3/accuracy.html)与[Time series cross-validation](https://otexts.com/fpp3/tscv.html)。本实验借鉴评价原则，不借用文献数据集的精度承诺。', '']
    (out / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
