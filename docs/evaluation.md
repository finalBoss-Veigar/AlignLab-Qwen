# 独立评估与证据边界

本次实现评估工程接口和开发测试，未加载模型，未执行完整 IFEval，未收集人工标注。没有模型胜率或后训练提升结论。`tests/test_evaluation.py` 中的规则输出仅验证聚合器，不能称为 IFEval 模型成绩。

## 统一生成协议

先冻结隔离的测试 prompt 清单，再用同规模原始 Instruct、继续 SFT、SFT+DPO 分别生成。1.5B 与 7B 必须分开报告。`evaluation.generate.generate_local` 只从本地目录加载真实 Transformers 模型，默认 dry-run；实际运行需要 `dry_run=False` 和已经安装的训练/推理环境，不自动下载权重。

每条生成记录包括 `prompt_id`、`prompt`、`response`、`model_id`、`model_revision`、`tokenizer_revision`、`chat_template_sha256`、`size_class`、`decoding`、`source_split` 和明确的 `synthetic` 布尔值。`sample_id` 可区分同一 prompt 的重复生成。实际 tokenizer 记录 prompt/response token 数；不以字符数冒充 token 数。输入超长时明确失败，不悄悄截断测试题。

本地真实生成输入还要求 `synthetic=false`、`dataset_id`、`revision`、`source_row_id`；只接受 UltraFeedback 的 `test_prefs` 或 google/IFEval 的评测专用 `train`/`ifeval`，fixture 不能通过改目录自动变成真实评测。生成记录输入指纹、实际设备、硬件、torch 版本和并发。`timeout_seconds` 通过每题间的截止检查和 Transformers `max_time` 控制软截止，超时结果失败且不评分；它不能抢占卡死的模型加载/CUDA 内核。需要严格强制截止时应从外部进程运行并使用操作系统超时。

比较器要求两组覆盖完全相同的 prompt/sample，prompt 正文、规模、tokenizer revision、聊天模板哈希、解码配置和测试来源一致。空回答保留为真实失败证据，不丢弃；空结果文件不能评分。当前默认贪心解码 `do_sample=false, max_new_tokens=512`。本地生成使用非流式接口，所以 TTFT 为 null。

## 人工盲评流程

`evaluation.blind.export_blind(left_rows, right_rows, output_dir, seed=42, limit=200)` 从 `test_prefs` prompt 按种子抽样，保留该 prompt 下的重复回答；synthetic 开发数据仅允许 `fixture_test`。按随机打乱的左右序列分配 A/B，左右总数之差不超过 1。候选文本不带模型标识；模型映射单独保存在 `private_key.json`，不要发给标注人，也不要提交 Git。程序不改写可能包含自称信息的回答，因此人工评估仍可能有内容层面的识别偏差，应记录在讨论中。

交给标注人的文件只有 `candidates.jsonl`、`annotations.jsonl`、`RUBRIC.md`。`label` 支持 A胜、B胜、平局、无法判断，亦支持 `A/B/tie/unjudgeable`。未评分留空。根据指令遵循、事实准确、相关性、有用性评分，不因为更长而直接判赢；无法核验时填无法判断，不能猜。failure_tags 支持重复、漏指令、错误拒答、事实错误对应的英文标记。

`import_blind(candidates, private_key, annotations)` 校验候选文件指纹、ID 唯一性、标签合法性，恢复左侧模型视角。分母 `N=W+L+T`，纯胜率 `W/N`，含平局得分 `(W+0.5T)/N`。无法判断和未评分分别计数，并从 N 排除；这可能产生非随机缺失，报告必须看排除率。N=0 时两项指标和置信区间为 null，不填 0。没有标注时只产生待评文件。

置信区间按 prompt cluster 重采样，保留同一 prompt 的所有回答和标签，不把它们当独立观测；默认种子 42、2000 次、95% percentile interval。只有一个 prompt 或全部同结果时区间可能退化，不能推导真实总体无不确定性。增加独立 prompt、报告样本量，核心结论在预算允许时复验随机种子。当前导入格式每个 case 只接受一份最终标签；多标注员应先制定一致的仲裁规则，不能重复导入使 N 膨胀。

报告按 prompt token 长度、左侧回答 token 长度、左右回答长度关系和 task_type 分桶。缺少 tokenizer 长度时明确记 `unknown_tokens`，不猜。损失和人工标记生成带 prompt_id/source/model_revision 的 bad case；无法判断和失败案例只能来自实际导入记录，不虚构对话。

## 官方 IFEval

2026-11-13 已核对一手资料：

- [官方 README，固定 commit](https://github.com/google-research/google-research/blob/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/instruction_following_eval/README.md)：命令入口和 JSONL 输入规范。
- [官方 evaluation_main.py](https://github.com/google-research/google-research/blob/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/instruction_following_eval/evaluation_main.py)、[evaluation_lib.py](https://github.com/google-research/google-research/blob/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/instruction_following_eval/evaluation_lib.py)：输出 strict/loose 的逐指令布尔结果与聚合定义。代码 Apache-2.0。
- [google/IFEval 数据卡，固定 revision](https://huggingface.co/datasets/google/IFEval/tree/966cd89545d6b6acfd7638bc708b98261ca58e84)：Apache-2.0，英文，当前 541 条，字段 `key,prompt,instruction_id_list,kwargs`。split 虽叫 `train`，在本项目只能用于最终评测，不能训练、调参或挑 checkpoint。

代码 pin 为 `f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81`，由 `git ls-remote` 确认；GitHub REST commit 查询遇到匿名限流，因此使用 git 协议核验。数据 pin 为 `966cd89545d6b6acfd7638bc708b98261ca58e84`，由 Hugging Face API 核验。没有克隆整个 google-research 大仓库。

在独立评估环境获取固定源码及其依赖后，从源码仓库根目录运行的官方命令是：

```sh
python -m instruction_following_eval.evaluation_main \
  --input_data=/absolute/path/official_inputs.jsonl \
  --input_response_data=/absolute/path/responses.jsonl \
  --output_dir=/absolute/path/fresh_output
```

依赖以固定源码 `instruction_following_eval/requirements.txt` 为准；其中 `absl` 的 Python 发行包名通常为 `absl-py`，另有 langdetect、nltk、immutabledict。NLTK 的资源依赖也需在正式环境逐项验证。轻量开发环境不安装这些包。不要为了凑形式上的指标用自制规则替代官方实现。

封装 `evaluation.ifeval.run_official(repo_dir,input_path,response_path,output_dir,dry_run=True)` 检查输入字段、完整覆盖和重复 prompt，dry-run 返回“未执行”。真实执行要求固定 git HEAD 且评估源码无修改。HF 展平的 kwargs 中新增 null 列会去除再交给官方接口；不改变有效 kwargs 或 prompt 正文。response JSONL 必须包含 `prompt` 和 `response`。实际运行日志、输入/回答指纹、耗时、退出状态写入新输出目录。非零退出、超时和解析失败均保留失败记录并向调用者抛错。

`generate` 生成的 `generations.jsonl` 可以直接作为 `--responses`，无需另写转换脚本；官方读取器只取 `prompt/response`，额外的模型和长度字段仍保留在原生成文件中。完成显式数据下载、模型本地生成及官方源码安装后，在项目根目录和独立评估环境运行：

```sh
# 默认只检查格式，不运行 IFEval；确认环境后在末尾加 --execute。
python -m alignlab ifeval --repo vendor/google-research --input data/raw/public/eval_prompts/ifeval_input.jsonl --responses runs/experiments/base-ifeval/generations.jsonl --output runs/experiments/base-ifeval-score
# 实际成功评分后，把生成的 metrics.json 汇入实验报告。
python -m alignlab report --output runs/experiments/base-report --ifeval runs/experiments/base-ifeval-score/metrics.json
```

这些是下一阶段的真实命令，当前没有上述真实生成文件或官方评估结果。没有执行时不要运行第二条并伪造输入。三个模型各使用新的生成/评分目录，但 input 文件、tokenizer、聊天模板、解码与输出上限保持相同。

聚合 `eval_results_strict.jsonl` 与 `eval_results_loose.jsonl` 的 `follow_all_instructions`、`follow_instruction_list`，检查覆盖和一致性，分别输出 strict/loose 的 prompt-level 与 instruction-level。本模块没有实现一套“近似 IFEval”规则。

真实运行还与固定官方仓库的 `data/input_data.jsonl` 比较完整规范化内容：只有题目、指令、参数全部匹配才标 `full_official_dataset`，子集或自造题明确标 `subset_or_custom_development_inputs`。直接调用聚合器时 `source_verified=false`，不能冒充已经核验并执行官方源码。

## 报告、工程指标与边界

`evaluation.report.write_report` 生成 UTF-8 JSON、适合 Excel 打开的 CSV 和独立 HTML。流程测试要求 `fixture=True` 且路径包含 `fixtures`；真实实验不得写入 fixtures。测试报告大标题固定显示 **SYNTHETIC FIXTURE · 流程测试（不是模型成绩）**。空评估可正常出报告，但所有未测量指标为 null 并写原因。

奖励准确率/奖励 margin 属于 DPO 训练诊断，必须连同 reference revision 保留，与生成盲评胜率分栏；不能比较不同 reference 的隐式 reward。吞吐需要固定硬件、输入/输出长度、并发、实际耗时，全部记录齐全后才接受数值。TTFT 只接受真实流式首 token 计时；不能用总延迟替代。

本次没有接入任何 LLM judge 或收费 API。未来接入需用户显式配置服务，再记录裁判模型/版本、完整提示词、互换顺序一致性和成本，且只能称代理指标。
