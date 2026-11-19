# Synthetic fixture（仅流程测试）

这里所有对话都是人工编写的微型软件测试输入，不来自 UltraChat / UltraFeedback，不是训练语料或模型生成成绩。

包括合法样本、重复/近重复 prompt、与隔离评测重合的 prompt、空回答、非法角色顺序、相同偏好回答、上下文不一致的偏好对。预期过滤原因写入 `rejected.jsonl`。

`FixtureTokenizer` 是离线流程测试用分词器；它的长度不是 Qwen token 长度。正式处理必须显式提供实际模型的、锁定 revision 的本地 tokenizer。
