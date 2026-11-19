# 数据治理与防泄漏

本次执行的是 `data/fixtures/` 的 **synthetic fixture 流程测试**。没有下载或处理完整 UltraChat / UltraFeedback，2 万 SFT / 1 万 DPO 只是训练子集预算。`data_manifest.json` 和 `*_quality_report.json` 给出每一步实际计数；不足不会复制样本补齐。

## 数据与入口

- `alignlab.data.schema`：SFT 使用 `messages`；DPO 内部保存共享 `messages` 上下文（最后一条 user）以及 chosen / rejected 两个 assistant 对象。UltraFeedback 原字段是两个完整对话列表，转入时比较 `chosen[:-1] == rejected[:-1]`，只取末尾回答，防止重复拼接 prompt。
- `alignlab.data.pipeline.run_fixture_pipeline(project_root, output_dir=None)`：离线测试全流程，返回 manifest。默认写到新的 `data/processed/fixture-时间戳/`。
- `alignlab.data.pipeline.run_pipeline(sft_path, dpo_path, eval_path, output_dir, ...)`：真实数据入口，需要实际 tokenizer、锁定 revision、来源元数据与全量评测排除清单。输出目录非空会失败，原始 JSON/JSONL 只读。
- `alignlab.data.sources.download_sources(output_dir, allow_download=True, sft_limit=None, dpo_limit=None)`：显式下载数据的可选入口，需要单独安装 `datasets`。不会下载模型；本次未执行。可限制训练来源池的下载条数，但所有官方测试 split 与 IFEval 排除清单必须完整扫描。源行编号使用原 split 下标。

下载器还在同一次遍历生成 `eval_prompts/test_prefs.jsonl` 与 `eval_prompts/ifeval.jsonl`，包含模型生成入口需要的平铺来源字段、唯一 prompt_id、prompt 与 synthetic=false。IFEval 保留 `key/instruction_id_list/kwargs`，并单独生成官方评测器格式的 `eval_prompts/ifeval_input.jsonl`。这些 prompt 不含参考答案；偏好生成上下文只取 chosen[:-1]。三个模型使用同一个文件/固定抽样清单，不能各自重新采样。

公开卡确认 UltraChat `train_sft` 有 `prompt` / `prompt_id` / `messages`；UltraFeedback `train_prefs` 有 `prompt` / `prompt_id` / `chosen` / `rejected` / `messages` / `score_chosen` / `score_rejected`。本项目不重新推测、排序或修改源数据的 chosen/rejected。两者以英语任务为主，不能当中文客服能力成绩。

每条内部记录保留 `dataset_id / revision / source_split / source_row_id / source_file / synthetic`、`prompt_hash`、`cleaning_reasons` 与 `group_id`。删除项写 `rejected.jsonl`，预算排除可通过 `split_assignments.jsonl` 与最终内部文件对应，所有文件指纹写 manifest。

## 隔离顺序

1. 校验角色交替、非空字符串、合法最终角色、偏好共同上下文和不同有效回答。去除明显控制字符与退化重复文本。`quality_filter=False` 仅关闭额外质量规则，基本合法性、长度安全和防泄漏始终保留。
2. 将 SFT、DPO 和评测排除清单联合分组。NFKC + casefold + 空白规范化只用于哈希/匹配，保留原文。对全部 user turn（包括裁剪前 turn）使用 5 字符 n-gram 集合 Jaccard，默认阈值 0.85，可配置。
3. 精确重复和近重复边求连通分量。与官方评测同组的全部训练来源样本删除；因此传递相似链也会保守排除。SFT / DPO 之间允许同一训练 prompt 的来源复用，但共享 split。
4. 每个任务每组保留首条，固定种子按组划分 train / validation，最后无放回抽取训练预算。验证子集从训练来源中产生。官方测试与 IFEval 不参与调参。`assert_no_leakage` 检查组交集。

真实排除清单必须包含 UltraChat `test_sft/test_gen`，UltraFeedback `test_sft/test_gen/test_prefs` 以及 `google/IFEval:train`。后者虽名为 train，仍仅作独立评测。`evaluation.jsonl.meta.json` 记录完整遍历、每个 split 实际条数和文件 SHA256，formal 流程校验这些证据，漏项会失败。人工提供清单的“完整”标记是输入契约；只有已实际遍历全部数据的下载产物才应声明完整。

近重复采用全局稀有度排序的前缀倒排索引产生候选，再以完整 n-gram 集合验证，没有近似搜索或候选截断。大量长对话仍会占内存，正式跑之前应先限训练来源池做资源试测并记录范围；不能把局部扫描报告称作全量数据治理。

## Token 长度与监督

真实数据只接受实际 tokenizer，调用 `apply_chat_template(..., tokenize=True, add_generation_prompt=False)` 统计包含聊天模板的完整序列 token。`load_tokenizer(path, revision)` 只从本地加载，禁止隐式联网或远程代码。

超过长度时只删除最老的完整 user+assistant turn，保留 system 与最终 user/assistant；不切断角色、不修改回答正文。最后一轮仍超长则删除，原因 `overlong_final_turn`。DPO 的共同上下文同时裁剪，两个完整回答均保留，绝不截断掉偏好差异。保留原始长度、处理后长度、chosen/rejected 回答 token 数与差值。回答 token 数是正文 `encode` 统计；训练实际 response mask 另由框架 token/label 检查验证。

SFT 至少含一个非空 assistant 内容 token。`fixture_token_labels` 仅测试基本 mask 不变量，明确标为 synthetic；它不是 Qwen / LLaMA-Factory 真实 labels 的证明。正式训练必须使用固定 LF 预处理器打印真实 token/label 样本并验证 user/system/padding 屏蔽。

## 格式与报告

导出 ShareGPT formatting + OpenAI role/content tags；DPO registry 设置 `ranking: true`、chosen/rejected 映射，两候选是 assistant 对象。四个名称：`alignlab_sft_train`、`alignlab_sft_validation`、`alignlab_dpo_train`、`alignlab_dpo_validation`。每条数据实际执行往返检查。

HTML 数据报告记录原始、过滤、评测重合、去重、划分、预算排除与最终条数，以及 token 长度分布、来源、文字脚本启发式、任务类别和偏好长度差。启发式语言概览不是精确语言鉴定。无样本时分布为空并使用 null，不制造成绩。
