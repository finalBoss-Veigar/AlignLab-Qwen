# 本地推理与边界

轻量环境可以启动无权重的 HTTP 服务：`python -m alignlab serve`。默认 127.0.0.1:8000；GET /health 为 200 表示进程存在，GET /ready 为 503 表示无模型，POST /generate 也返回 503，不提供 mock 回答。`--dry-run` 仅检查命令，不启动服务。测试使用临时本地端口、真实 HTTP 请求和 3 秒超时。

真实 Transformers 推理需在独立训练环境安装兼容依赖，显式指定 `--model 本地完整权重目录 --revision 对应模型revision或导出指纹`。只读本地权重，禁止自动下载；使用模型自己的聊天模板，贪心解码，输入最大 1024 tokens、输出最大 256 tokens。HTTP body 上限 64KiB，单请求串行模型执行；忙时 429，输入错 400，模型未就绪 503。生成最长约 60 秒（Transformers 在迭代边界检测），客户端默认 70 秒超时。

客户端：`python -m alignlab request --prompt 'Explain gradient accumulation briefly.'`。调用代码位于 src/alignlab/serving/client.py。无权重时预期明确失败，不应把这种状态当作推理验证成功。

吞吐只在固定硬件、输入/输出 token 数、并发=1 时报告；单次服务结果记录实际 tokens、耗时和 tokens/s。未实现流式时间测量，TTFT 一律为 null，不能用总时长代替。跨模型对照用相同 generation 配置，保留模板哈希、revision、输入集合指纹。

## 可选 Linux vLLM

本次未安装、未运行 vLLM。它和训练环境完全隔离；在 Linux NVIDIA 主机新建 `.venv-vllm`，安装 requirements/serve-vllm.txt（候选固定版 0.8.5），先检查该版驱动/PyTorch/CUDA要求。当前只支持完整、非量化合并模型目录，不承诺 4-bit 权重或任意 adapter 可直接服务。

`bash scripts/serve_vllm.sh /absolute/path/to/merged_model` 默认 dry-run；加 `--execute` 才启动。脚本只监听 localhost，HF_HUB_OFFLINE=1。使用独立 8001 端口的 OpenAI compatible `/v1/chat/completions`，请求应带客户端超时和 max_tokens。脚本未在本机 Linux GPU 实测。

官方命令参考：[vLLM 0.8.5 OpenAI compatible server](https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html)，访问日期 2026-11-13。
