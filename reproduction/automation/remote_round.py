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
import time

# Local controller supplies PAYLOAD, SECONDS_LEFT and CODE_COMMIT.
_started = time.monotonic()
_finished = threading.Event()
def watchdog():
    if not _finished.wait(SECONDS_LEFT):
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
            logs=sorted(Path('/content/drive/MyDrive/colab-llm-experiments/jobs').glob(ROUND_ID+'/job.log'))
            if logs:
                print('PROGRESS:', logs[0].read_text()[-500:], flush=True)
        except Exception:
            pass
threading.Thread(target=heartbeat, daemon=True).start()

from google.colab import drive
drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/colab-llm-experiments')
out = root/'public_subset_eval'
out.mkdir(parents=True, exist_ok=True)
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
(jobdir/'provenance.json').write_text(json.dumps({'code_commit':CODE_COMMIT,'payload_sha256':PAYLOAD_SHA256,
    'protocol':'v1, unchanged prompts/NF4/2048/128/greedy','round_id':ROUND_ID},indent=2))
# Installation and evaluation use subprocesses so loaded kernel imports cannot cache old versions.
with (jobdir/'setup.log').open('a') as log:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(automation/'requirements-colab.txt')],
                   stdout=log,stderr=subprocess.STDOUT,check=True,
                   timeout=max(1,SECONDS_LEFT-120-(time.monotonic()-_started)))
remaining=max(1,SECONDS_LEFT-120-(time.monotonic()-_started))
print('DEPENDENCIES READY; full evaluation starting',flush=True)
result = subprocess.run([sys.executable,str(automation/'job.py'),'--state-dir',str(jobdir),
    '--timeout-minutes',str(remaining/60),'--',sys.executable,'-u',str(automation/'evaluate.py'),
    '--out',str(out),'--run-name','qwen3_8b_nf4_ctx2048_v1','--full'])
print('EVALUATION EXIT:',result.returncode,flush=True)
for suffix in ('_metrics.json',):
    path=out/('qwen3_8b_nf4_ctx2048_v1'+suffix)
    if path.exists():
        print(path.read_text(),flush=True)
print('ROUND_LOG:',str(jobdir),flush=True)
# Successful write-back precedes local-controller teardown. A failure still triggers CLI cleanup.
drive.flush_and_unmount()
_finished.set()
if result.returncode:
    raise RuntimeError(f'Evaluation exit {result.returncode}; inspect saved job state')
