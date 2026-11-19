# 训练环境与启动边界

源码校验的 SHA256 仅将 CRLF 换行为 LF 后计算，规则记录为 `source_hash_normalization: CRLF_to_LF_only`；兼容 Windows Git autocrlf 与 Linux checkout，不忽略其他字节修改。数据、checkpoint 和导出权重的哈希仍使用原始字节，不采用此源码专用规则。

本次已实现配置生成、固定版本字段检查、启动/导出、断点指纹、运行期审计入口；**未安装 GPU 训练依赖、未下载权重、未执行真实训练**。本机 RTX3060 Laptop 6GB 只验收开发流水线；7B 候选在 Linux NVIDIA 24GB 上试跑，1.5B 单独记分。

固定 LLaMA-Factory v0.9.3 commit `ca75f1edf3cb50343ed1c98605141c3e22075b5f`，Apache-2.0，setup.py 要求 Python>=3.9；优先独立 Python3.11，当前轻量开发使用已有3.12。官方参数与案例在 vendor/LlamaFactory；upstream_schema.json 从官方 dataclass AST 采集 init 字段、源文件/行号/哈希，另采集 Transformers4.51.3（commit `b99fa8251c7327130243a0b76e36d0a922492e41`）TrainingArguments。静态字段检查不等于运行期官方解析器/GPU通过。

Linux 项目根目录的训练安装示例（下一阶段主动执行；可能下载大量依赖，但没有权重下载命令）：

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/train-constraints.txt -e './vendor/LlamaFactory[bitsandbytes]'
python -m pip install -e .
python -m pip check
python -m pip freeze > runs/train-environment-resolved.txt
```

train-constraints.txt 是官方范围内的核心依赖候选约束，尚未完成 Linux 安装验证，不能称完整 CUDA lock；实际解析后需保存完整 freeze 和驱动版本。macOS 不装 CUDA/bitsandbytes；vLLM 应独立环境，不套用训练锁。若 vendor 不随复制迁移，先在项目根目录 git clone 官方仓库到 vendor/LlamaFactory，再 git checkout 上述 commit，并恢复 vendor/transformers-4.51.3 两个参数源文件；静态验证必须通过源码 SHA256。

配置说明：debug_1_5b（5步）/sft_7b/dpo_7b/dpo_1_5b/export_7b/export_1_5b/export_dpo_7b/export_dpo_1_5b。SFT 导出以原始高精度 Instruct 为底座；DPO 导出以高精度 SFT merged 为底座并加载 DPO adapter，生成 dpo-merged，供 Transformers/vLLM 本地验证。导出记录实际 stage，不把 DPO 产物伪装成 SFT。SFT 学习率1e-4，DPO5e-6，rank16/alpha32/target all，4bit NF4 double quant，关闭packing，batch1，梯度累积16，梯度检查点启用；1.5B调试长度512、累积4。其余长度1024。它们都是起始候选。默认BF16，真实启动检查设备支持；不支持时需明确改为fp16并记录，不能两个都开。

生产路径 data/processed/production，数据登记名 alignlab_sft_train、alignlab_sft_validation、alignlab_dpo_train、alignlab_dpo_validation。先用与训练一致 tokenizer/template、cutoff_len 处理正式数据，再启动。fixture 明确禁止用于生产训练。输入数据manifest、配置指纹、代码版本与环境都随run保存。有效批量=微批次×梯度累积×WORLD_SIZE，GPU总数单独记录。当前启动封装限单进程单卡；不把机器GPU总数误当训练进程数。

使用 root CLI 的 train/export 命令查看 `--help`，默认 dry-run；只有显式执行选项才真正调用官方训练。正式启动强制 HF_HUB_OFFLINE 与 TRANSFORMERS_OFFLINE，因此缺权重会报错，不会突然下载几十GB。必须先由用户决定资源位置，准备指定revision的缓存模型；DPO模型必须来自带manifest的本项目SFT高精度导出。

继续训练必须提供完整 checkpoint：adapter/model、optimizer.pt、scheduler.pt、rng_state*.pth、trainer_state.json、alignlab_fingerprint.json。数据与配置指纹不一致拒绝恢复；callback 在每次保存时写指纹，故异常退出前已保存的checkpoint也可核对。单独加载adapter只能算新训练初始化。当前不支持 DeepSpeed/FSDP分片状态恢复，不能绕过上述检查。

训练退出码、完整process.log、runtime_audit.json、实际config、train_results、trainer_state与产物地址保存在runs/experiments唯一目录。dry-run在runs/checks；peak_vram_bytes来自训练进程 torch.cuda.max_memory_allocated，独立审计进程另算，不是整机reserved峰值；耗时包括封装与审计。OOM属于失败run，保留日志；更改长度、批量或模型另建配置，不声称训练已完成。
