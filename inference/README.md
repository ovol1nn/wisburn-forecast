# 模型1历史输入候选版

这是用于离线研究及只读影子运行的候选模型，不是已发布的控制模型。它不替换Profile、不连接DCS、不输出控制指令。模型2不变。

## 输入与输出

- 输入：最近20分钟、120行、10秒间隔的已清洗历史数据。列为`timestamp`和本包`metadata.json`中的304个特征；物理单位必须与训练一致。
- 这不是Silver原始信号清洗入口。输入必须沿用既有质量规则；现场确认异常的负值继续筛除，禁止为凑齐数据放宽筛选。在线清洗只能用当时可见历史，不能中心插值或引用未来样本。
- 必须声明预测起点`--as-of`，且它等于历史最后时间；多余未来行、错乱间隔、重复列会拒绝。接口没有未来真实工况参数。
- 输出：60行未来预测、14个目标，均为物理单位；同时保留原预测、目标状态、缺失信息、短时修正原因。JSON中的`null`代表不可用，不能解释为0。
- 缺少辅助列/非有限输入按训练均值兼容，但明确报告，不声称恢复信息。当前目标缺失时该目标不可用；目标历史缺失超过20%、总体输入有效单元不足50%或模型输出非有限时，退回该目标的当前值并标明降级。20%/50%是工程兜底阈值，不是已标定的置信度。
- 时间戳和输入范围检查无法证明上游物理单位、信号含义或清洗正确，调用方仍需核验。

## 短时修正

只作用于一次风预热器出口温度`B3TE03`。依据近5分钟范围与近10分钟净变化判断历史是否平稳，再对最短时距向当前值收缩。权重随时距连续回归原模型，180秒及以后完全不变；其余13个目标不变。若温度历史缺失，禁用该修正。

实际是否启用以`policy.json`为准。参数在较早校准日期确定，较晚日期只复核；现有数据此前已被查看，因此这些结果不是新的最终测试。完整验收记录保存在候选目录的`evaluation/`和`acceptance.json`。

2026-10-22氧量补充实验未通过替换验收：`oxygen_correction.py`仅供报告脚本复算，不被`predict.py`调用，也没有进入已冻结v1包。不能因为源码中存在校准函数就视为氧量效果已改善。研究结果位于forecast artifacts的`candidates/model1_oxygen_corrected_v2_20261022/analysis_report.md`；该目录只有实验产物，没有新的模型bundle。

## 直接运行

先进入生成包的`bundle`目录，使用已经装有PyTorch、NumPy、Pandas和einops的Python环境。初次准备环境可执行`python -m pip install -r requirements.txt`；CPU与GPU的PyTorch安装方式应与机器环境匹配，示例默认CPU，不需要8张卡。

打开`example_origin.txt`查看示例起点，然后执行（把起点替换为该文件中的时间）：

```text
python predict.py --history example_history.csv --as-of "示例起点时间" --output prediction_demo.json
```

更换真实输入时：

```text
python predict.py --history your_cleaned_history.csv --as-of "预测起点时间" --output prediction_new.json
```

输出文件必须是新文件，不覆盖已有结果。可以使用`--device cuda:0`在单卡执行；不需要启动训练。程序会校验本包资产和源码哈希，任何不匹配直接报错；不要手改本包参数后仍沿用原验收记录。

## Python调用

在此包目录中的独立Python进程内：

```python
import pandas as pd
from predict import HistoryOnlyPredictor

predictor = HistoryOnlyPredictor('.')
history = pd.read_csv('your_cleaned_history.csv')
result = predictor.predict(history, as_of='预测起点时间')
```

保持模型实例复用，避免每次重新加载权重。TSLib使用`models/layers/utils`绝对模块名，若宿主已导入其他同名模块，程序会拒绝，需在独立进程运行；本候选包不修改主工程的导入结构。

## 验收边界

先核对JSON的`target_status`和`quality`，不能只取数值。即使状态正常，`usable_for_control`也始终为`false`：未验证新日期、现场单位/时序、在线因果清洗、端到端延迟与控制安全边界前，不作为自动控制依据。预热温度的保守改进不能代替其他目标的独立验收，也不能保证未来新工况效果。

源代码测试：在forecast模块上级项目中执行`python -m unittest discover -s offline/forecast/tests -p test_history_only_inference.py`。打包与评价入口为`reports/build_history_only_candidate.py`，正式模型资产和既有部署配置不会被改写。
