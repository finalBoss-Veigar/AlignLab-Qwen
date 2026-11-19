# 11月18日晚环境恢复证据

无卡实例上的环境准备，不是GPU训练或模型效果实验。代码起点dc8fc33；手动传入下载及导入检查脚本，执行时工作区不干净。后续提交记录修复与验证，不能把提交时间倒填成运行版本。

- 本机`python -m pytest -q`：81 passed in 3.91s；包括4项下载完整性检查，使用合成字节，不是7B性能测试。
- 云端`python scripts/check_training_imports.py`：training-imports.json记录实际版本、UTC时间、通过的NumPy/Arrow/Datasets转换及cuda_available=false。
- `pip check`第一次成功仍未发现二进制兼容问题；真实导入报`pyarrow requires NumPy 2.0 or newer, found 1.26.4`。固定PyArrow19.0.1后，实际导入与转换及pip check通过。
- 原始HuggingFace链路失败，hf-mirror大文件跳转停滞。ModelScope仅作为文件传输；模型仍锁定HF revision a09a35458c702b33eeacc393d103063234e8bc28。每个文件大小与LFS SHA256/普通文件Git blob都须匹配，最终以完整manifest为准。
- 最终manifest于2026-11-18T15:22:54Z记录completed，14文件/4权重分片共15,242,807,270字节，用时1402.094秒；该时间包含下载与校验，不是模型训练。model-download-completed.json已取回、解析并确认文件数量；未运行模型生成或训练。

以下原始文件已SCP取回本机`runs/cloud_20261118/`，远端与本地SHA256一致；大日志、安装列表和数据不放入Git。

| 文件 | SHA256 |
|---|---|
| training-imports.json | 3f769294f842c666ff67eabed2d6ce3e55e141902b5f138d7bf55e60de8dd312 |
| environment-repair-20261118.log | 6589bcee66ec037ca738ecabaf2e7d85ae21b1040be57a7e5db309f973ef9e9f |
| train-environment-resolved.txt | ada37a4ccdea6bfd90b8eacb06b2fddb299f1b4721098035dd9813a4eb30be02 |
| trial-launch-remote-original.log | 52290b62175898530a22aa4ebfed7e53c082f3259dbfca0a52bf7d6c62af578a |
| model-download-completed.json | 52fe06dd986b495f257c111170902a049fdc96e2f38b24903f31efe436013308 |
| model_download_alternate_20261118.log | 20842b5bfc4eb53ff4af0f03e3b3450e06cdaf3fce837079c3fe9475adc13b1f |

学习入口：先读check_training_imports.py理解“安装成功”和“实际可导入”的区别；再读fetch_model.py的verify_file，亲自运行`python -m pytest tests/test_model_download_integrity.py -q`，观察正确文件通过、截断或哈希不符被拒绝。这些检查不证明掌握QLoRA或DPO，也不计入尚未完成的训练验收分数。
