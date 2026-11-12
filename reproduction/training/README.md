# 训练代码分析与无外部 API 验证

本入口是对作者公开脚本的数据流与双任务训练的可检查实现，复用其任务提示；不加载已发布 CoPoLLM LoRA，不导入不稳定的 TRL/Unsloth 接口。
这是原数据控制组，不是自己的 PRO+DSCO，也不声称已经验证 CPRL 的实际疗效。

## 已复现的源码问题

- `task4/sft_unified.py:289-304`：history 已含当前 user，又 append 一次。真实 train.json 的 17060 个生成样本全部重复当前发言。修复为历史去掉最后当前 user 后再渲染一次。
- `sft_unified.py:113-119, 975-983`：掩码失败回退为全序列损失；备用 collator 没屏蔽 padding。新入口明确构造 prompt/target IDs，前缀与 padding 的 label 均为 -100，真实 EOS 参与监督；无法构造则排除并记录。
- `sft_unified.py:721,929`：默认 max_steps=120 覆盖 3 epochs；标准 HF 分支未将 max_length 传入 SFTConfig。新入口显式 optimizer 步数与长度，正式 epochs 训练尚未实现。
- `task1/create_dqn_dataset.py:294`：字符串 bias_tags 直接 join 会逐字分隔；状态构造需先 normalize_tags。
- `task2/run_online_training.py:323-329`：context 不含当前患者发言，中文冒号判断后又按英文冒号切分。新状态契约优先 current_utterance 或 s_text 的当前发言，不回退为上一轮/“你好”。
- `run_online_training.py:358`：成功交互后固定同一低危/轻微 next-state；须对新患者回复重新标注和嵌入。新 transition 契约强制传入真实 next annotation，缺失即报错。
- `run_online_training.py:253,258-267,381`：文件/嵌入失败被随机向量、零向量掩盖。新契约拒绝维度错误、非有限数和未标注状态。
- `task2/config.yaml:11`：默认关闭症状奖励；reward evaluator 实际质量项评的是 skill/coherence，不能当临床症状改善测量。奖励数值、KL 方向与论文一致性仍须逐项核对。
- `task3/generate_sft_dataset.py:37`：存在测试集改写路径；本训练入口只接受训练文件，测试集不进入数据构造或验证。

## CPU 检查

`python reproduction/training/test_data.py`：无泄漏、无重复当前 user、按整场对话分割、重复对话同一分区、response-only/EOS mask、next-state 变化和失败拒绝。
`uv run --python 3.13 --with torch==2.11.0 --with numpy python reproduction/training/test_dqn.py`：直接提取作者 QNetwork/ReplayBuffer/optimize_model，使用显式合成单元测试数据验证更新、终止状态与存取。此类测试不构成 CPRL 真实策略收益证据。

## GPU 小规模检查

`python3 reproduction/automation/train_round.py --minutes 30 --units 5 --gpus A100 --high-mem`

仅做 generation-only 和原数据联合训练各20步；joint 严格分类/生成 1:1 交替。microbatch=1，accumulation=1，rank16/alpha32/dropout.05，lr2e-4，NF4+BF16，最大1024，GC开启。
这些参数是管线调试条件，不能替代正式预算一致的消融实验。

训练/验证按对话分成1212/134场；使用预先冻结的每任务16条小训练集、每任务16条验证集。只看验证集 loss 和8条分类输出；最终测试集不用于20步调试选参。

第10步检查 adapter、optimizer、scheduler、RNG 序列化与同进程恢复；不宣称冷启动恢复已测。
GPU 输出按16MB分块拉回本地并校验 SHA-256/ZIP CRC，避免新GPU的Drive授权阻塞；本轮没有自动写入Drive。
完成/失败/预算到限均停止独立训练会话，并核验后端endpoint消失。原先101.23分钟的评测预算与这轮新批准的30分钟训练预算分开记录。
