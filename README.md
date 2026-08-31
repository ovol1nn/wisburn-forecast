# 兴蓉三号炉预测模型模块

本目录用于训练和评估焚烧炉预测模型。模型根据最近 20 分钟的历史工况，预测未来 10 分钟的关键指标变化，为后端的工况判断、人工规则、MPC 和安全裁决提供依据。

模型只回答“未来工况可能怎样变化”，不直接生成控制指令，不连接或写入 DCS。是否形成建议仍由人工规则、MPC 和安全裁决共同决定。

## 当前模型

| 模型 | 关注范围 | 输入 | 输出 | 预测目标 |
| --- | --- | --- | --- | --- |
| `model1` | 炉膛燃烧和蒸汽工况 | 过去 20 分钟：120 个 10 秒点、300 个特征 | 未来 10 分钟：60 个 10 秒点 | 主蒸汽流量、省煤器出口氧量、炉膛氧量、中上部炉膛温度、垃圾层厚度 |
| `model2` | 烟气净化和排放 | 过去 20 分钟：1200 个 1 秒点、109 个特征 | 未来 10 分钟：600 个 1 秒点 | CO、HCl、NOx、SO₂、烟尘 |

两模型的原始预测结果的时间粒度不同（10秒和1秒），但当前后端会把两模型的结果统一到 **60 个十秒节点** 后再进入工况判断和控制链。

模型 2 的原始 600 个逐秒预测不会丢弃，保留给审计和离线排查。

已完成的工作包括：训练数据构造、特征/目标/时间窗口校验、checkpoint 与 scaler 校验、ONNX 数值对齐，以及各目标 1、5、10 分钟预测效果的离线验收。已发布模型位于 `assets/models/xingrong_3/`（没有在此仓库下）；本目录只负责后续训练和改进。

## 当前最值得优化的问题

### 1. 两个模型时间步长不统一

模型 1 是 `120 → 60` 的十秒预测，模型 2 是 `1200 → 600` 的逐秒预测。两者虽然都覆盖“过去 20 分钟、未来 10 分钟”，但模型 2 需要在进入控制链前再聚合为十秒点，训练、展示和控制的时间网格不完全一致。

这不一定意味着模型 2 必须立刻改成十秒模型：排放指标可能确实包含更快的波动信息。但它是当前最重要的对比优化课题。建议在相同 Silver 数据和相同时间切分上，额外训练一个模型 2 的十秒版本（`120 → 60`），与现有逐秒版本比较：

- 1、5、10 分钟各目标误差和趋势方向；
- 十秒控制节点上的误差，而不是只比较逐秒误差；
- 模型推理稳定性、训练成本和 ONNX 部署效果；
- CO、NOx、SO₂ 等快速变化时，逐秒模型是否真的比十秒模型更早、更可靠地给出有效趋势。

只有当逐秒版本在这些比较中有明确收益时，才保留不同时间步长；否则统一为十秒网格会让训练、模型联动和控制解释更清晰。任何改变都必须创建新版本，不能覆盖当前发布模型。

### 2. 在已有数据中更充分挖掘规律

优先做以下四项，不要一开始盲目增加点位或更换复杂模型：

1. **按工况分别评估。** 将历史窗口按负荷、炉温、氧量、垃圾层厚度、配风状态、排放水平和启停状态分组。找出哪些典型工况预测最差，再针对这些工况增加训练权重或单独分析，而不是只看全体平均误差。
2. **学习“变化过程”，不只学习单点数值。** 对每个目标同时评估未来 1、5、10 分钟值、上升/下降方向和十分钟终点变化。现有模型 1 已有趋势与终点损失，可系统比较其权重；模型 2 也应检查污染物的持续上升、回落和超限前趋势是否被提前捕捉。
3. **检查真正有效的输入信息。** 利用现有 300/109 个特征和已有派生特征，分析哪些历史点位、滞后时间和组合状态在不同工况下对目标最有帮助。先处理缺失、时间错位、长期常值和停炉段，再比较删减无效特征、补充合理滞后特征或改进派生特征。
4. **用控制相关场景验收。** 除常规误差外，抽取“氧量下降且 CO 上升”“SO₂/NOx 持续上升”“主蒸汽与炉温变化”等窗口，检查预测是否足够早且方向稳定，能否为后续规则和 MPC 提供有用信息。

每次实验只改变少量因素，并保留数据版本、代码版本、配置、随机种子、逐目标对比和失败案例。这样才能判断改进来自模型本身，还是来自数据口径变化。

## 仓库文件说明

```text
offline/forecast/
├─ data_engineering/  Silver → 清洗、重采样、训练窗口和时间切分
├─ training/          训练入口、数据桥接、训练规范
├─ tslib/             训练使用的 Crossformer / TSLib 源码
├─ contracts/         特征、目标和时间窗口契约
├─ reports/           离线预测效果与训练报告
├─ inputs/            外部 Silver 数据和点位表说明
└─ tests/             路径与契约测试
```

训练产生的大文件统一写入 `offline/artifacts/forecast/`，包括重采样缓存、训练集、checkpoint、ONNX、预测数组和训练输出。它们不会上传 Git。后端运行时只读取验收后发布到 `assets/models/xingrong_3/` 的模型 bundle。

## 训练数据存放

Silver 数据和点位分类表应放在仓库外的受控位置，例如：

```text
D:\wisburn-data\xingrong_3\
├─ silver\by_point\        # 每个点位一份 Parquet，例如 B3_F80.PV.parquet
├─ point_classification.xlsx
└─ SHA256SUMS.txt           # 推荐：数据包校验值
```

不要将 Silver Parquet、原始 CSV、checkpoint、ONNX、预测数组或压缩数据包复制到本目录或提交到 Git。单独使用本模块时，在 PowerShell 中设置本机路径：

```powershell
$env:WISBURN_FORECAST_POINT_ROOT = "D:\wisburn-data\xingrong_3\silver\by_point"
$env:WISBURN_FORECAST_CLASSIFICATION_XLSX = "D:\wisburn-data\xingrong_3\point_classification.xlsx"
$env:WISBURN_FORECAST_ARTIFACT_ROOT = "D:\wisburn-artifacts\forecast"
```

这些变量只指定本机文件路径，不涉及账号、密钥或 DCS。

## 复现训练

1. 安装与 GPU/CUDA 匹配的 PyTorch，并安装 `training/requirements_vm_training.txt`。
2. 准备 Silver 数据和点位分类表，设置上述三个环境变量。
3. 检查环境和路径：

   ```powershell
   python offline/forecast/data_engineering/build_model_datasets.py check-env
   ```

4. 在 `data_engineering/build_model_datasets.py` 顶部选择模型和时间参数，构造分块数据；再运行 `data_engineering/prepare_dataset_splits.py` 生成按时间顺序的训练、验证、测试切分。
5. 运行 `training/run_chunked_tslib_training.py`。第一次保持 `INTERACTIVE_RUN_TRAINING = False`，检查 metadata、目标、特征数和训练命令；确认后改为 `True` 才启动训练。
6. 对新版本完成 ONNX 对齐及逐目标 1、5、10 分钟验收，再由主工程构建发布 bundle。训练通过不等于自动替换当前运行模型。

完整契约见 [contracts/README.md](contracts/README.md) 和 [training/TRAINING_SPEC.md](training/TRAINING_SPEC.md)。

## GitHub 与协作

创建私有仓库 `wisburn-forecast` 时，不要勾选 GitHub 的 README 或 `.gitignore`，因为本目录已提供。创建空仓库后：

```powershell
Set-Location "D:\code\pythonprojects\智慧焚烧\wisburn-platform\offline\forecast"
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "feat: add xingrong3 forecast training module"
git remote add origin https://github.com/<你的账号>/wisburn-forecast.git
git push -u origin main
```

提交前确认暂存区中没有 `.parquet`、`.pth`、`.onnx`、`.npy`、`.zip`、虚拟环境或 `offline/artifacts/` 内容。数据和训练产物通过受控共享盘单独交付，并附数据版本与校验值。

给合作者的要求很简单：只在本模块内修改数据构造、训练、评估和说明；从个人分支提交 Pull Request；每次实验说明数据版本、改动、逐目标结果和失败案例；不提交数据/权重。
