> 当前连接配置与实际命令请优先阅读 [CONNECTED_WORKFLOW.md](CONNECTED_WORKFLOW.md)。下文保留通用流程。

# Colab Pro + SSH + GitHub + Drive

## 当前准备情况

本地仓库在本次检查时没有 commit、没有 remote。新增工具尚未在远端 GPU 执行。
不要直接 git add .：目录还包含上游源码、论文摘录、模型元数据，先只纳入复现代码。

## 首次建立 GitHub 同步

先创建一个空的私有 GitHub 仓库，获得仓库 URL。下列命令在本机项目目录执行，将 REPOSITORY_URL 替换为实际地址：

```bash
git add .gitignore reproduction/automation reproduction/CoPoLLM_SSH_Launcher.ipynb reproduction/CoPoLLM_Qwen3_8B_Colab_start.ipynb
git diff --cached --stat
git commit -m "Add resumable Colab evaluation workflow"
git remote add origin REPOSITORY_URL
git push -u origin main
```

`.gitignore` 不会自动剥离 Notebook 内嵌的输出和秘密；推送前检查暂存内容。
本次新增启动 Notebook 没有输出或凭据。SSH 私钥、HF/GitHub Token 不进 Git。

## 每次申请 GPU 后

1. 用有 Pro 且计算单元余额为正的账号在 Colab 选择 GPU 并连接。普通 Pro 的 GPU 分配和重连不保证自动化。
2. 在 Notebook 中挂载同账号 Drive。SSH 连接和 Drive 挂载可能需要每次重建，不把临时 IP 当永久地址。
3. 在 SSH 会话中克隆代码到 Colab 本地盘，避免从 Drive 执行大量小文件：

```bash
cd /content
git clone REPOSITORY_URL cbt
cd cbt
git checkout COMMIT_SHA
```

以后更新时，确认没有远端未保存修改，再执行 `git fetch origin` 和 `git checkout COMMIT_SHA`。COMMIT_SHA 必须替换为本轮完整提交号。
本地 Codex 修改 → 本地提交/push → Colab fetch/checkout；正在运行的实验期间不要更换代码。

4. 打开 `CoPoLLM_SSH_Launcher.ipynb`，安装依赖并重启会话、重新挂载 Drive，再运行任务格。
启动器由 Notebook 保持执行，SSH 用于排错/日志。Notebook finally 负责写回 Drive 并释放运行时。

## 同步此前进度，先使用 CPU

挂载 Drive 后，在 SSH 或 Colab 运行：

```bash
python reproduction/automation/inspect_results.py /content/drive/MyDrive/CoPoLLM/public_subset_eval/qwen3_8b_nf4_ctx2048_v1_diagnosis.jsonl
```

无须 GPU、torch 或模型。它输出唯一条数、重复键、上限数和原文。旧 NameError 不再依赖内存变量。
把当前 v1 的 diagnosis、metrics、manifest、revisions、pip-freeze 和 sample_manifest 打包下载到本地的 `runs/`，保留旧文件。
先核对每组唯一 10 条，再更新 PROGRESS.md。GitHub 只同步摘要，原始预测不推送。

## SSH 执行与自动停止

推荐 Notebook 启动器，因为 SSH 进程通常不能完整调用 Notebook 的释放接口。
若临时直接从 SSH 启动：

```bash
python reproduction/automation/job.py --state-dir /content/drive/MyDrive/CoPoLLM/public_subset_eval/jobs/review_v1 --timeout-minutes 60 -- python -u reproduction/automation/evaluate.py --out /content/drive/MyDrive/CoPoLLM/public_subset_eval --run-name qwen3_8b_nf4_ctx2048_v1 --limit 10
```

日志为 STATE/job.log；状态为 STATE/status.json。手动停止在前台按 Ctrl+C，或向 status.json 中的 supervisor_pid 发送 SIGTERM。
停止子进程仅停止计算，GPU 资源仍在。完成后在 Notebook 执行：

```python
from google.colab import drive, runtime
drive.flush_and_unmount()
runtime.unassign()
```

卸载写回失败时不自动释放，以便恢复；强杀或服务中断时自动释放不能保证，应到 Colab 手动“断开并删除运行时”。
不要把 torch.cuda.empty_cache() 当成释放计费运行时，也不要把断开 SSH 当释放 GPU。

## 全量评测的门槛

先调查基座 100% 达到输出上限；训练集划出验证集，冻结模板/解码/输出长度。
当前 CLI 保留原 v1 协议，不自动替你选择提示或修复输出。
`--full` 显式运行完整 981 条；更改协议前修改评测脚本并创建新 run-name。

evaluate.py 从起步 Notebook 导出，保持原提示、量化、两个条件及指标，移除了交互挂载和单条演示。
它逐条 flush/fsync、跳过已完成键、拒绝重复/损坏记录，并沿用模型 revision 与配置核验。
单个运行目录应固定一个 STATE；不要用不同 STATE 同时写同一预测文件。
运行时强制回收仍可能产生不完整末行，工具会报错，不会自动删除数据。
脚本尚未 GPU 实测，须先以现有 10 条对齐指标，不能直接宣称已验证整个训练流程。

后续训练需要额外实现 optimizer/scheduler/scaler/RNG 的完整断点，而不只是保存 LoRA 权重。
当前自动化只覆盖现有分类评测，不会启动 CPRL、生成 PRO 或训练 DSCO。

## 官方依据

Colab 资源与付费 SSH 限制：https://research.google.com/colaboratory/faq.html
官方释放接口源码：https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py
