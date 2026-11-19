# DPO 参考终点与必须执行的审计

本项目从 **已经是 Instruct 的 Qwen2.5** 继续 SFT。DPO 策略和参考必须都从同一次 SFT 结束的模型出发，不能把原始 Instruct 当参考。本次只有源码核对与离线检查，真实张量检查尚未执行。

采用单基座路线：原始高精度 Instruct + SFT adapter → 官方 export_model 非量化合并 → 保存完整 SFT 权重与 SHA256 清单 → 用 bitsandbytes NF4 重新加载该 SFT 基座 → 新建零增量 LoRA → DPO。export 配置禁止 quantization_bit/export_quantization_bit，官方导出器也会拒绝向量化模型合并 adapter。导出后重新量化会带来数值变化；DPO 策略与参考共享重新量化后的同一 SFT 基座，因此两者一致，但不等同于宣称与高精度 SFT 逐位一致。

固定源码是 LLaMA-Factory v0.9.3 / `ca75f1edf3cb50343ed1c98605141c3e22075b5f`：

- [create_ref_model](../vendor/LlamaFactory/src/llamafactory/train/trainer_utils.py) 的 LoRA 分支返回 None。
- [CustomDPOTrainer.compute_reference_log_probs](../vendor/LlamaFactory/src/llamafactory/train/dpo/trainer.py) 对同一模型进入 `disable_adapter()` 与 `torch.no_grad()`。这里关闭的是**新 DPO adapter**，底座已含 SFT，故参考没有退回原始 Instruct。
- [PairwiseDatasetProcessor](../vendor/LlamaFactory/src/llamafactory/data/processor/pairwise.py) 对共同上下文使用同一截断长度，response[0] 是 chosen，response[1] 是 rejected；prompt 标签为 -100。
- [PairwiseDataCollatorWithPadding](../vendor/LlamaFactory/src/llamafactory/data/collator.py) 将批次排列为所有 chosen 然后所有 rejected；padding 标签为 -100。
- 标准 sigmoid DPO 调用官方 trainer 的 **response token log probability 总和**，不是长度平均。只有 IPO/ORPO/SimPO 走平均路径；本项目拒绝这些替换。pref_ftx 与 label smoothing 为零，beta 起点 0.1。

`training/audit.py` 在真实训练之前使用官方 tokenizer、processor、collator、CustomDPOTrainer 执行以下运行期检查：

1. 打印真实 tokenizer 的 input_ids/labels/监督文本；独立比较 system/user/assistant 预期 mask，检查 padding。另检查实际第一条训练样本。
2. 对 DPO 验证 chosen/rejected 截断后仍不同、首步策略/参考 response log probability 一致（rtol/atol 1e-4）。
3. 所有非 LoRA 参数必须冻结，计算完整冻结参数 SHA256；用官方损失做一个**丢弃的审计步**，再验证参数哈希不变、参考 log probability 逐位相同。它是审计，不是替代 DPO 损失。
4. 审计独立子进程退出，训练重新加载干净终点；审计变化不会保存为训练初始权重。审计日志与 runtime_audit.json 随 run 保存。

这些检查需要实际 GPU、依赖、缓存权重和 production 数据。CPU fixture 标签测试仅验证检查器，不能宣称已验证真实 Qwen 训练。正式训练若审计失败会停止并保存日志；恢复训练时审计仍针对新建初始 adapter，真正恢复由官方 trainer 加载完整 checkpoint，不再要求已训练 adapter 与参考相同。

7B/1024/单卡24GB是待试验资源目标，不保证所有 DPO 配置装入显存。共享基座减少一份模型内存，但激活、双回答、梯度和优化器仍占显存。减少长度/累积策略/改用1.5B都要新建配置与记录，不静默换模型。
