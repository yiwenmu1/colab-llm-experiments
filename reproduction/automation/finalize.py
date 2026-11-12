"""Wait for an owned GPU round, verify results, generate and archive the report."""
import argparse,datetime,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent
p=argparse.ArgumentParser();p.add_argument('round_dir',type=Path);a=p.parse_args()
round_dir=a.round_dir.resolve();status=round_dir/'report_status.json'
def run(cmd):subprocess.run(cmd,check=True)
try:
    # Extra wall-clock guard; never allocate a GPU here.
    deadline=time.monotonic()+2*3600
    while not (round_dir/'final.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Round finalization wait expired')
        time.sleep(15)
    final=json.loads((round_dir/'final.json').read_text())
    if not final['release_verified']:raise RuntimeError('GPU release not verified')
    results=round_dir/'results/public_subset_eval';name='qwen3_8b_nf4_ctx2048_v1'
    preds=results/(name+'_diagnosis.jsonl');metrics=results/(name+'_metrics.json')
    rows=[json.loads(l) for l in preds.read_text().splitlines() if l.strip()]
    if len(rows)!=1962 or len({(r['key'],r['condition']) for r in rows})!=1962:raise RuntimeError('Incomplete unique prediction coverage')
    run([sys.executable,str(HERE/'recompute_metrics.py'),str(preds),'--reference',str(metrics),'--out',str(results/(name+'_metrics_recomputed.json'))])
    run(['uv','run','--python','3.13','--with','numpy','--with','scikit-learn==1.6.1',
         'python',str(HERE/'verify_sklearn.py'),str(preds),str(metrics)])
    report=PROJECT/'reproduction/reports/CoPoLLM_Qwen3_8B_public_subset_report_2026-11-11.md'
    run([sys.executable,str(HERE/'report.py'),str(results),'--out',str(report)])
    leases=[];total=0
    for f in sorted((Path.home()/'.config/colab-cli/history').glob('eval-20261111T*.jsonl')):
        events=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        starts=[e for e in events if e.get('event_type')=='session_created']
        ends=[e for e in events if e.get('event_type')=='session_terminated']
        if starts:
            if not ends:raise RuntimeError('An owned session lacks a termination record')
            mins=(datetime.datetime.fromisoformat(ends[-1]['timestamp'])-datetime.datetime.fromisoformat(starts[0]['timestamp'])).total_seconds()/60
            total+=mins;leases.append((f.stem,mins))
    if total>120:raise RuntimeError('GPU lease ledger exceeded approved time')
    usage=subprocess.check_output(['colab','usage'],text=True)
    content=report.read_text()+'\n## 自动执行与预算记录\n\n'
    content+='本轮先完成资源与持久化调试，再用单块 A100 80GB、四个互不重叠的 batch=1 进程完成剩余评测。CPU 线程数设为 1。所有失败和调试租用计入总占用。\n\n'
    content+='| 会话 | GPU 租用分钟 |\n|---|---:|\n'
    for session,mins in leases:content+=f'| {session} | {mins:.2f} |\n'
    content+=f'\n累计租用 **{total:.2f} 分钟**，上限 120 分钟。外部教师/模拟器 API 费用为 0。\n\n'
    content+='以下为完成后 CLI 显示值；账户余额差包含并行 CPU 会话且可能存在结算延迟。\n\n```text\n'+usage+'```\n'
    content+='\n已确认全部本轮 GPU 会话释放；保留原有 llm CPU 会话作为 Drive 持久化入口。完整原始预测与软件记录放本地/Drive，不推送 GitHub。\n'
    content+='\n## 下一轮范围\n\n先从干净基座分别检查原数据 generation-only 与分类＋生成联合 SFT（各 20 步），验证 response-only masking、两任务 loss 和断点恢复。该阶段可不调用外部 API。CPRL、PRO 和 DSCO 的独立重训仍需单独规划，不能用本报告替代。\n'
    report.write_text(content)
    original=Path('/Users/yiwenmu/Documents/ChatGPT/cbt/reproduction/reports');original.mkdir(parents=True,exist_ok=True)
    shutil.copy2(report,original/report.name)
    archive=original/'CoPoLLM_Qwen3_8B_public_subset_results_2026-11-11.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(report,report.name)
        for f in results.iterdir():
            if f.is_file() and f.suffix in ('.json','.jsonl','.txt'):z.write(f,'results/'+f.name)
        for folder in (PROJECT/'runs').glob('eval-*'):
            for name2 in ('budget.json','final.json'):
                f=folder/name2
                if f.exists():z.write(f,'rounds/'+folder.name+'/'+name2)
            for f in (folder/'results/jobs').rglob('*.json'):
                z.write(f,'rounds/'+folder.name+'/'+str(f.relative_to(folder/'results')))
    drive='/content/drive/MyDrive/colab-llm-experiments'
    for f in (preds,metrics,results/'checksums.json'):
        run(['colab','upload','-s','llm',str(f),drive+'/public_subset_eval/'+f.name])
    for f in (report,archive):run(['colab','upload','-s','llm',str(f),drive+'/'+f.name])
    status.write_text(json.dumps({'state':'complete','report':str(original/report.name),'archive':str(archive),
        'gpu_minutes':round(total,2),'gpu_release_verified':True,'predictions':1962},indent=2))
    print(status.read_text(),flush=True)
except BaseException as exc:
    status.write_text(json.dumps({'state':'failed','error':repr(exc)},indent=2))
    raise
