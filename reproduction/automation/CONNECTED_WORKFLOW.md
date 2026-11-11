# 已连接环境（2026-11-11）

GitHub: git@github.com:yiwenmu1/colab-llm-experiments.git
SSH: colab-gpu；CLI session: llm
远端项目: /content/llm-project
持久化根: /content/drive/MyDrive/colab-llm-experiments
旧实验根: /content/drive/MyDrive/CoPoLLM/public_subset_eval

首次检查 SSH 和 Git remote 正常、仓库初始提交 dbf3c79、当时 GPU 为 T4。
随后同名 llm 会话变成 CPU，项目与 Drive 挂载消失。当前不能假设旧目录或 GPU 仍存在。
历史 Notebook 的 A100 40GB 是上轮环境，不代表当前硬件。

## 自动申请、同步、停止

本机安装 google-colab-cli 0.8.0，官方支持以下命令。勿在已有 llm 会话时重复 new。

```bash
colab sessions
# 仅在 llm 已结束后申请；资源不足时报错即停止，不无限重试或自动升级更贵硬件。
colab new -s llm --gpu T4
colab drivemount -s llm
```

确认 SSH alias 仍指向 llm。新会话项目目录会消失，重新 git clone 到 /content/llm-project，固定本轮 commit；安装 requirements-colab.txt。
以上申请命令会消耗 Pro 计算单元；这里只准备命令，不自动新建额外会话。

```bash
ssh colab-gpu 'cd /content/llm-project && git pull --ff-only origin main'
# 先 CPU 审计新目录中归档的旧结果，不需要加载模型。
ssh colab-gpu 'cd /content/llm-project && python reproduction/automation/inspect_results.py /content/drive/MyDrive/colab-llm-experiments/public_subset_eval/qwen3_8b_nf4_ctx2048_v1_diagnosis.jsonl'
# 本地执行，默认不释放会话，适合交互排错
bash reproduction/automation/run_remote.sh 60
# 明确结束整个会话：任务完成/超时/失败后 flush Drive，再由本地 CLI 释放
bash reproduction/automation/run_remote.sh --release 60
```

run_remote.sh 的清理依赖本机脚本持续运行；电脑睡眠、断网、进程被强杀时不能保证自动 stop。
当前远端 job 有执行时限，但只停止 Python 不会释放 GPU。长时无人值守优先 Notebook 启动器的 finally，或另行部署可靠控制机。
不要同时启动 Notebook 启动器和 SSH 脚本；同一输出文件固定同一个 STATE。
评测完成续跑会跳过已有 10 条并重算指标，不会自动修复基座上限问题。

## 同步旧进度

旧目录保留不动，新目录中只复制不存在的文件（不覆盖已存在的新结果）。
同步完成后先 CPU 审计，再把摘要写进 PROGRESS.md。
原始数据/权重/预测不进 Git；只推代码和汇总记录。

## 本地修改的工作目录

当前 /Users/yiwenmu/Documents/ChatGPT/cbt 是材料目录，自己的 Git 没有历史，不要与远端强行合并。
在其 `runs/github-checkout` 下建立远端仓库的独立克隆，作为后续 GitHub 同步工作区。
本次新增工具同步到远端 /content/llm-project 后正常提交推送，保留远端既有历史。

官方 CLI：https://github.com/googlecolab/google-colab-cli

## 已授权自动轮次（120 分钟 / 25 CU）

`python3 reproduction/automation/auto_round.py` 自动申请独立 A100 会话，失败尝试一次 T4，执行冻结 v1 的 981×2 分类评测。
不会干扰既有 llm CPU 会话，不调用外部教师 API。
本地每 30 秒检查账户余额和消耗率，预留至少 1 CU / 2 分钟余量；远端另有独立期限保护。
完成、失败或预算到限后，控制器调用官方 stop 并查询后端确认具体 endpoint 消失。
控制器日志和 final.json 在 runs/eval-时间戳；远端日志与状态在 Drive/jobs/同一轮次目录。
远端强制释放、网络异常、Drive FUSE 延迟仍可能影响最后几条记录的持久化；完整 JSONL 可续跑，损坏末行必须备份核查。
这是两组公开子集分类评测，保留 v1 的基座达到上限问题，不宣称完整训练方法或论文原测试集复现。

新 GPU 运行时可能要求新的 Drive 浏览器授权。自动轮次因此改用 CPU 持久化桥：
- 从已有 llm CPU 挂载的 Drive 读取旧结果；GPU 只使用本地临时盘。
- GPU 每 15 秒生成完整行快照；控制器每约 45 秒下载到本机，并上传快照 ZIP 至 CPU 的 Drive 挂载。
- 自动轮次无需将 Drive 凭据授予每个新 GPU VM。GPU 端故障最多可能损失尚未同步的一小段新预测，已同步记录可续跑。
- 结束前再次拉取快照，再释放 GPU。最终结果需从快照归档回 public_subset_eval 并核验完整性。
