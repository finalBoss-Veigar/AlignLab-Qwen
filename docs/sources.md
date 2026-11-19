# 一手资料、固定版本与许可证

访问日期均为 **2026-11-13（Asia/Shanghai）**。机器可直接访问本机桌面。只下载了小型数据卡/metadata及代码源码；未下载模型权重或正式数据分片。机器可读记录见 sources.lock.json；数据卡原始快照见 source_snapshots。许可证来自相应模型/数据卡或代码LICENSE；公开数据仍需关注上游生成来源和许可适用范围。

| 资料与用途 | 实际锁定 revision/commit | 许可证 |
|---|---|---|
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)：唯一训练框架，README安装范围、训练器与导出实现 | v0.9.3，`ca75f1edf3cb50343ed1c98605141c3e22075b5f` | Apache-2.0 |
| [数据格式README](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)：ShareGPT messages、ranking/chosen/rejected映射；实际读取固定tag的data/README.md | 同上，[固定链接](https://github.com/hiyouga/LlamaFactory/blob/ca75f1edf3cb50343ed1c98605141c3e22075b5f/data/README.md) | Apache-2.0 |
| [examples](https://github.com/hiyouga/LlamaFactory/tree/main/examples)：train_qlora/train_lora及merge_lora范例；实际读本地固定checkout | 同上，[固定目录](https://github.com/hiyouga/LlamaFactory/tree/ca75f1edf3cb50343ed1c98605141c3e22075b5f/examples) | Apache-2.0 |
| [Alignment Handbook](https://github.com/huggingface/alignment-handbook)：只参考SFT→偏好优化的实验组织和数据配方，不混入安装栈 | `53c11c328684f965162f41ff93185bb9e0448263` | Apache-2.0 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)：正式模型卡及revision | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)：独立调试模型 | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)：train_sft、字段及测试排除集合 | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT |
| [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)：train_prefs完整chosen/rejected消息列表与测试隔离 | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT |
| [IFEval官方实现README](https://github.com/google-research/google-research/blob/master/instruction_following_eval/README.md)：官方evaluation_main及strict/loose返回格式 | `f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81` | Apache-2.0 |
| [google/IFEval](https://huggingface.co/datasets/google/IFEval)：官方数据协议，key/prompt/instruction_id_list/kwargs | `966cd89545d6b6acfd7638bc708b98261ca58e84` | Apache-2.0 |

补充字段来源：Transformers 4.51.3，commit `b99fa8251c7327130243a0b76e36d0a922492e41`，Apache-2.0。LLaMA-Factory的model/data/finetuning参数和Transformers TrainingArguments通过AST生成336字段快照，保留源文件哈希与行号，见 src/alignlab/training/upstream_schema.json。它验证字段存在与本项目安全约束，**不能替代安装GPU依赖后的官方parser和运行验收**。

实际字段核对：UltraChat含prompt、prompt_id、messages；UltraFeedback含prompt、prompt_id、完整chosen/rejected、messages、score_chosen、score_rejected。两者卡片均以英语为主。IFEval在HF使用名为train的split，但项目视为独立评测，禁止用于训练或调参。

网络记录：GitHub网页examples一次访问失败，后通过固定tag本地checkout核对；GitHub API出现403限流，git ls-remote与git clone成功；urllib复查时遇到连接重置，保留首次成功获取的数据卡和metadata，没有换成猜测revision。Alignment Handbook通过固定checkout核对README/LICENSE。没有因为网络受阻改动技术方案。

可选服务参考：[vLLM 0.8.5官方文档](https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html)，Apache-2.0，固定发行版0.8.5；仅检查服务命令文档，未安装/运行，详情docs/serving.md。
