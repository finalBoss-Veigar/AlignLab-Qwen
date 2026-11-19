# 架构与产物流动

公开数据卡与 revision → 显式下载只读 data/raw → 内部 messages / prompt+chosen+rejected → 合法性过滤 → 所有训练/评测 prompt 的精确和近重复分组 → 排除官方评测污染 → 按组划分 train/validation → 预算采样 → tokenizer 长度约束 → LLaMA-Factory 格式与 dataset_info.json。

原始 Qwen2.5-Instruct → QLoRA 继续 SFT adapter → 原始高精度基座合并 SFT adapter → 非量化 SFT 合并模型 → 重新 4-bit 加载、新 DPO adapter → DPO 导出与本地推理。每一步记录来源，不把原始 Instruct 当 SFT 参考。

原始 Instruct / SFT / SFT+DPO 使用同一隔离测试 prompt、同一 tokenizer/chat template、同一解码设置 → generations.jsonl → 官方 IFEval / 随机平衡盲评 → prompt 级 bootstrap / 长度分桶 / 可追溯 bad cases → metrics.json、summary.csv、HTML。

| 模块 | 入口与责任 |
|---|---|
| CLI | src/alignlab/cli.py：路由与错误码 |
| 路径/配置 | paths.py、config.py：桌面发现、项目根、配置与指纹 |
| 数据 | data/schema.py、cleaning.py、dedup.py、pipeline.py、formats.py、sources.py：内部协议、治理、数据产物 |
| 训练 | training/configs.py、runner.py：固定框架字段、候选配置、dry-run、训练/导出、失败保留 |
| 跟踪 | tracking/manifest.py：独立 run ID、环境、配置、代码状态、退出结果 |
| 评估 | evaluation/generation.py、blind.py、bootstrap.py、ifeval.py、report.py |
| 推理 | serving/server.py、client.py、evaluation/generate.py：本地真实权重、HTTP 边界和生成文件 |

`runs/fixtures` 只放流程测试；`runs/experiments` 放真实实验。训练封装独立日志目录以实际 run manifest 为准。raw 与模型只读，新的 processed、adapter、合并权重各用独立目录。`.venv` 不含训练依赖。固定源码位于 vendor（不提交）；字段快照和版本锁属于项目证据。
