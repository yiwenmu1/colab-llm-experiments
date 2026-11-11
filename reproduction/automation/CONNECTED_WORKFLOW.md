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
