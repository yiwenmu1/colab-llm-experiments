"""Colab kernel entry point; rendered by auto_round.py with payload/deadline globals."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import zipfile
import time

# Local controller supplies PAYLOAD, SECONDS_LEFT and CODE_COMMIT.
_started = time.monotonic()
_finished = threading.Event()
def watchdog():
    time.sleep(SECONDS_LEFT)
    if True:
        print('REMOTE DEADLINE: releasing runtime', flush=True)
        try:
            subprocess.run(['sync'], timeout=10)
        finally:
            from google.colab import runtime
            runtime.unassign()
threading.Thread(target=watchdog, daemon=True).start()

def heartbeat():
    while not _finished.wait(15):
        print('ROUND heartbeat; elapsed seconds:', round(time.monotonic()-_started), flush=True)
        try:
            logs=sorted(Path('/content/copollm-output/jobs').glob(ROUND_ID+'/job.log'))
            if logs:
                print('PROGRESS:', logs[0].read_text()[-500:], flush=True)
        except Exception:
            pass
threading.Thread(target=heartbeat, daemon=True).start()

root = Path('/content/copollm-output')
out = root/'public_subset_eval'
out.mkdir(parents=True, exist_ok=True)
for name,content in SEED.items():
    (out/name).write_bytes(base64.b64decode(content))
project = Path('/content/copollm-round')
project.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as archive:
    for member in archive.getmembers():
        target = (project/member.name).resolve()
        if not target.is_relative_to(project.resolve()) or not member.isfile():
            raise ValueError('Invalid payload member')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
automation=project/'reproduction/automation'
jobdir=root/'jobs'/ROUND_ID
jobdir.mkdir(parents=True, exist_ok=True)
_snapshot_lock=threading.Lock()
def _write_snapshot():
    tmp=Path('/content/copollm-snapshot.tmp')
    with zipfile.ZipFile(tmp,'w') as z:
        for folder in (out,jobdir):
            for f in folder.iterdir():
                if f.is_file() and f.suffix in ('.json','.jsonl','.txt','.log'):
                    data=f.read_bytes()
                    if f.suffix=='.jsonl':
                        data=data[:data.rfind(b'\n')+1]
                    z.writestr(str(f.relative_to(root)),data)
    tmp.replace('/content/copollm-snapshot.zip')
def snapshot():
    with _snapshot_lock:
        _write_snapshot()
snapshot()
def snapshot_loop():
    while not _finished.wait(15):
        try: snapshot()
        except Exception as exc: print('SNAPSHOT warning:',repr(exc),flush=True)
threading.Thread(target=snapshot_loop,daemon=True).start()
(jobdir/'provenance.json').write_text(json.dumps({'code_commit':CODE_COMMIT,'payload_sha256':PAYLOAD_SHA256,
    'protocol':'v1, unchanged prompts/NF4/2048/128/greedy','round_id':ROUND_ID},indent=2))
# Installation and evaluation use subprocesses so loaded kernel imports cannot cache old versions.
with (jobdir/'setup.log').open('a') as log:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(automation/'requirements-colab.txt')],
                   stdout=log,stderr=subprocess.STDOUT,check=True,
                   timeout=max(1,SECONDS_LEFT-120-(time.monotonic()-_started)))
env_code = '''import json,sys,torch,importlib.metadata as m
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
'packages':{k:m.version(k) for k in ['transformers','peft','accelerate','bitsandbytes','huggingface_hub','scikit-learn']}},indent=2))'''
(jobdir/'environment.json').write_text(subprocess.check_output([sys.executable,'-c',env_code],text=True))
remaining=max(1,SECONDS_LEFT-120-(time.monotonic()-_started))
print('DEPENDENCIES READY; full evaluation starting',flush=True)
result = subprocess.run([sys.executable,str(automation/'job.py'),'--state-dir',str(jobdir),
    '--timeout-minutes',str(remaining/60),'--',sys.executable,'-u',str(automation/'evaluate.py'),
    '--out',str(out),'--run-name','qwen3_8b_nf4_ctx2048_v1','--full','--allow-ephemeral-output'])
print('EVALUATION EXIT:',result.returncode,flush=True)
for suffix in ('_metrics.json',):
    path=out/('qwen3_8b_nf4_ctx2048_v1'+suffix)
    if path.exists():
        print(path.read_text(),flush=True)
print('ROUND_LOG:',str(jobdir),flush=True)
# Successful write-back precedes local-controller teardown. A failure still triggers CLI cleanup.
snapshot()
_finished.set()
# Independent of the controller: leave a short final-snapshot retrieval window, then release.
def release_completed_vm():
    from google.colab import runtime
    runtime.unassign()
_release_timer=threading.Timer(90,release_completed_vm)
_release_timer.daemon=True
_release_timer.start()
if result.returncode:
    raise RuntimeError(f'Evaluation exit {result.returncode}; inspect saved job state')
