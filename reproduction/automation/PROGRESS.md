# 复现进度快照（2026-11-11）

证据为共享 Notebook 保存输出，尚未读取实际 Drive 预测文件：
https://colab.research.google.com/drive/182UHTEUncT9c_WCqLdDBUIjUMPqGSjw8

- 已完成：A100 40GB；CUDA 13；bitsandbytes 0.49.0；作者 LoRA 和 Qwen3-8B NF4 加载；单条分类和单条生成检查。
- 数据：训练对话 1346；测试对话 324；有效诊断 981；排除 41；高危 42。
- 小样本：base / copollm 各 10 条；Macro F1 0.1559066 / 0.2321429；Micro F1 0.2777778 / 0.3870968。
- 两组 HRMDR=0，但只有 2 个高危样本。
- 阻塞全量比较的事项：基座 10/10 达到 128-token 上限；作者 LoRA 0/10。必须先读取原始输出，不能凭低 F1 或上限标志判定原因。
- 末尾 NameError 是会话变量未恢复，不能据此判定已保存文件丢失。
- 未完成：训练/验证按对话划分、协议验证、完整 981×2 分类评测、成批生成质量评测、任何自主训练和 CPRL/PRO/DSCO。

下一轮顺序：CPU 审计已有预测 → 训练集划验证集 → 验证集确定输出协议 → 固定代码/config/revision → 全量预测与重算指标 → 原数据两组 SFT 20-step 检查。

代码、配置、摘要进 GitHub；原始预测、对话、模型、训练检查点放 Drive。更换协议须更换 run-name；禁止将不同条件的结果混入 v1。
