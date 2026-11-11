"""GPU kernel bootstrap, frozen v1 protocol with four independent batch=1 shards."""
import base64,io,json,os,signal,subprocess,sys,tarfile,threading,time,zipfile
from pathlib import Path
_started=time.monotonic()
_finished=threading.Event()
_snapshot_lock=threading.Lock()
root=Path('/content/copollm-output');out=root/'public_subset_eval'
project=Path('/content/copollm-round')
out.mkdir(parents=True,exist_ok=True);project.mkdir(exist_ok=True)
for name,content in SEED.items():(out/name).write_bytes(base64.b64decode(content))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:
    for member in archive.getmembers():
        target=(project/member.name).resolve()
        if not target.is_relative_to(project.resolve()) or not member.isfile():raise ValueError('Invalid payload')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
automation=project/'reproduction/automation'
jobdir=root/'jobs'/ROUND_ID;jobdir.mkdir(parents=True,exist_ok=True)

def snapshot():
    with _snapshot_lock:
        tmp=Path('/content/copollm-snapshot.tmp')
        with zipfile.ZipFile(tmp,'w') as z:
            for f in root.rglob('*'):
                if f.is_file() and f.suffix in ('.json','.jsonl','.txt','.log'):
                    data=f.read_bytes()
                    if f.suffix=='.jsonl':data=data[:data.rfind(b'\n')+1]
                    z.writestr(str(f.relative_to(root)),data)
        tmp.replace('/content/copollm-snapshot.zip')

def hard_release():
    from google.colab import runtime
    runtime.unassign()

def watchdog():
    time.sleep(SECONDS_LEFT)
    print('REMOTE DEADLINE: releasing owned GPU',flush=True)
    try:subprocess.run(['sync'],timeout=10)
    finally:hard_release()
threading.Thread(target=watchdog,daemon=True).start()

def heartbeat():
    while not _finished.wait(15):
        try:
            snapshot()
            counts={}
            for f in out.glob('*_diagnosis.jsonl'):
                counts[f.name]=sum(1 for line in f.read_text().splitlines() if line.strip())
            print('PROGRESS',round(time.monotonic()-_started),counts,flush=True)
        except Exception as exc:print('SNAPSHOT warning',repr(exc),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
(jobdir/'provenance.json').write_text(json.dumps({'code_commit':CODE_COMMIT,'payload_sha256':PAYLOAD_SHA256,
    'protocol':'v1 unchanged prompts/NF4/2048/128/greedy/batch1','parallel_workers':4,'round_id':ROUND_ID},indent=2))
snapshot()
with (jobdir/'setup.log').open('a') as log:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(automation/'requirements-colab.txt')],
        stdout=log,stderr=subprocess.STDOUT,check=True,
        timeout=max(1,SECONDS_LEFT-180-(time.monotonic()-_started)))
env_code='''import json,sys,torch,importlib.metadata as m
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
'packages':{k:m.version(k) for k in ['transformers','peft','accelerate','bitsandbytes','huggingface_hub','scikit-learn']}},indent=2))'''
environment=json.loads(subprocess.check_output([sys.executable,'-c',env_code],text=True))
(jobdir/'environment.json').write_text(json.dumps(environment,indent=2))
workers=4 if environment['gpu_memory_bytes']>=38_000_000_000 else 1
base='qwen3_8b_nf4_ctx2048_v1'
keys={s['key']:i for i,s in enumerate(json.loads((out/'sample_manifest.json').read_text()))}
old=[json.loads(l) for l in (out/(base+'_diagnosis.jsonl')).read_text().splitlines() if l.strip()]
processes=[]
try:
    for i in range(workers):
        name=base+'_worker'+str(i)
        for suffix in ('_manifest.json','_revisions.json'):
            (out/(name+suffix)).write_bytes((out/(base+suffix)).read_bytes())
        (out/(name+'_diagnosis.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old if keys[r['key']]%workers==i))
        state=jobdir/('worker'+str(i));state.mkdir(exist_ok=True)
        remaining=max(1,SECONDS_LEFT-120-(time.monotonic()-_started))
        command=[sys.executable,str(automation/'job.py'),'--state-dir',str(state),'--timeout-minutes',str(remaining/60),'--',
            sys.executable,'-u',str(automation/'evaluate.py'),'--out',str(out),'--run-name',name,'--full',
            '--allow-ephemeral-output','--shard-id',str(i),'--num-shards',str(workers)]
        log=(state/'supervisor.log').open('a')
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc)
        # Stagger model initialization to limit peak VRAM and avoid concurrent repository checkouts.
        ready=out/(name+'_ready.json')
        while not ready.exists():
            if proc.poll() is not None:raise RuntimeError(f'Worker {i} failed during loading')
            if time.monotonic()-_started>SECONDS_LEFT-180:raise TimeoutError('Deadline during loading')
            time.sleep(2)
        print('WORKER READY',i,ready.read_text(),flush=True)
    while any(proc.poll() is None for proc in processes):
        if time.monotonic()-_started>SECONDS_LEFT-120:raise TimeoutError('Evaluation time budget reached')
        if any(proc.poll() not in (None,0) for proc in processes):raise RuntimeError('Worker failed; snapshots retained')
        time.sleep(2)
    if any(proc.returncode for proc in processes):raise RuntimeError('Worker nonzero exit')
    merged={}
    for f in out.glob(base+'_worker*_diagnosis.jsonl'):
        for line in f.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);k=(r['key'],r['condition'])
            if k in merged:raise RuntimeError('Duplicate shard key')
            merged[k]=r
    assert len(merged)==1962,len(merged)
    rows=sorted(merged.values(),key=lambda r:(keys[r['key']],r['condition']))
    (out/(base+'_diagnosis.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    subprocess.run([sys.executable,str(automation/'recompute_metrics.py'),str(out/(base+'_diagnosis.jsonl')),
                   '--out',str(out/(base+'_metrics.json'))],check=True)
    print('COMPLETE:1962', (out/(base+'_metrics.json')).read_text(),flush=True)
finally:
    for proc in processes:
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    for proc in processes:
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    snapshot();_finished.set()
    timer=threading.Timer(90,hard_release);timer.daemon=True;timer.start()
