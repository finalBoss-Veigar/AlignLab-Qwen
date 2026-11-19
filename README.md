# AlignLab · 继续后训练会改善指令遵循吗？

对已经会对话的 Qwen2.5-7B-Instruct 继续做 QLoRA 指令微调（SFT），再用成对偏好做 DPO。在相同的 541 道 IFEval 指令题上，原始模型整题通过 **386 道**，继续 SFT 后降到 **323 道**，主 DPO 配置回到 **333 道**。若用严格指令遵循选模型，这次实验支持保留原始 Instruct。[正式结果与逐题配对统计](docs/evidence/formal_20261119/summary.json)

| 单卡 A800 80GB，固定模板与贪心解码 | 严格整题通过 | 严格单项指令通过 |
|---|---:|---:|
| 原始 Qwen2.5-7B-Instruct | 386 / 541（71.35%） | 659 / 834 |
| 继续 SFT | 323 / 541（59.70%） | 568 / 834 |
| SFT 后 DPO，β = 0.1 | 333 / 541（61.55%） | 587 / 834 |
| SFT 后 DPO，β = 0.05 | 334 / 541（61.74%） | 592 / 834 |

一题可能包含多个要求，严格整题需要全部通过。主 DPO 相对 SFT 增加 1.85 个百分点，按题配对重采样的 95% 区间为 [−0.74, +4.25] 个百分点。四个条件都生成了固定的 200 道偏好题，人工盲评尚待完成。当前数字衡量的是英语指令遵循，训练使用一个种子。[评测方法](docs/evaluation.md) · [逐模型证据目录](docs/evidence/formal_20261119)

## 实验是怎样推进的

| 当时的问题 | 采取的行动与得到的结果 | 可核查的记录 |
|---|---|---|
| 训练集和独立评测如何隔离？ | 固定来源版本、清洗并按相近提示分组。保留 14,289 条 SFT 训练对话、7,400 对 DPO 训练偏好，分别留出验证数据。 | [数据流水线](src/alignlab/data/pipeline.py) · [各阶段规模与指纹](docs/evidence/stage_20261117/data_evidence.json) |
| 几个模型真的在回答同一题吗？ | 发现数据快照里有一道题与官方题面不同。保留旧输出，用官方题面重新生成该题，其余 540 条相同输入复用已有回答。 | [评测协议](docs/evidence/cloud_20261118/evaluation-protocol.json) · [基线核验](docs/evidence/formal_20261119/base_evidence.json) |
| 单卡训练可以复现和恢复吗？ | 先核验助手标签与冻结参考模型，在 checkpoint-100 实际恢复 SFT。完整 SFT 做了 893 步，两组 DPO 各做 462 步，再导出、重载并统一生成。 | [配置](configs/gpu_20261118) · [训练记录](docs/evidence/formal_20261119/training_evidence.json) · [恢复证据](docs/evidence/formal_20261119/recovery_evidence.json) |
| SFT 退化是输出上限造成的吗？ | 91 道从通过变失败的题中，仅 5 道的 SFT 回答触及 2,048 个输出 token 上限。有的答复保留了关键词，却少写了一处要求的 Markdown 高亮。 | [逐题长度诊断](docs/evidence/length_diagnostics_20261119/summary.json) · [诊断实现](scripts/analyze_length_diagnostics.py) |
| 能用简单规则删掉格式错误的训练示范吗？ | 在 33,184 个训练问答轮中，窄规则匹配了 6 轮，3 个回答未用项目符号。回读发现其中两题允许使用编号，答案确实使用了编号。这个规则会误删合规示范。 | [预先固定的检查方法](docs/experiment_plan.md) · [完整计数](docs/evidence/list_format_20261120.json) · [逐例复盘](HANDOFF.md) |

后续比较需要区分“必须用指定格式”“允许多种格式”“只是在讨论格式”三类开发题。这一步可以先比较已有模型，再决定是否值得更换训练数据或调整 SFT 学习率。现有的 541 道题已用于上述诊断。

## 先复算结果，再运行流程

在仓库根目录用 Python 3 读取已提交的官方评分汇总。只需标准库，预期输出依次是 base 386、sft 323、dpo 333、dpo_beta005 334，分母都是 541。

~~~shell
python -c "import json; m=json.load(open('docs/evidence/formal_20261119/summary.json', encoding='utf-8'))['models']; print({k: [v['strict']['passed_prompts'], v['strict']['n_prompts']] for k, v in m.items()})"
~~~

从新环境验证本地流水线，在项目根目录运行以下 PowerShell 命令。macOS/Linux 使用 `bash scripts/bootstrap.sh` 安装，并将解释器路径换为 `.venv/bin/python`。smoke 使用微型示例检查流程，正式训练的实测结果在上方证据表。[安装脚本](scripts/bootstrap.ps1) · [单卡训练配置](configs/gpu_20261118)

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
& .venv/Scripts/python.exe -m alignlab doctor
& .venv/Scripts/python.exe -m alignlab smoke
& .venv/Scripts/python.exe -m pytest -q
~~~

## 实现与复盘

项目复用 LLaMA-Factory v0.9.3 的模型加载、QLoRA、SFT/DPO 训练器和导出能力。仓库中的工程实现覆盖样本治理、配置和标签审计、冻结参考模型核验、生成溯源及逐题评估。开发和文档整理使用了 AI 辅助。数据与底座权重从固定来源取得，仓库提供配置、代码、聚合与逐题评分证据。[框架与模型来源](docs/sources.md) · [训练审计实现](src/alignlab/training/audit.py)


## 阅读与复现

[第一人称面试逐字稿 · PDF](docs/AlignLab-Interview-40min.pdf) · [可编辑讲稿](docs/INTERVIEW_40MIN.md) · [固定验收范围](docs/PROGRESS.md)

本公开版本取自开发快照 `45a6e12755804bfdee7532c9fd48bdd28a5c5271`。实验文件记录当时的代码版本、数据指纹与运行环境。公开仓库从本次发布开始记录提交。 数据证据中一处本机分词器路径已改为项目相对路径。
