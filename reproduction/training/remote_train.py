"""Kernel bootstrap for bounded, API-free real-data training checks."""
import base64,io,json,hashlib,os,signal,subprocess,sys,tarfile,threading,time,zipfile
from pathlib import Path
started=time.monotonic();finished=threading.Event();lock=threading.Lock()
project=Path('/content/copollm-training');out=Path('/content/training-output')
project.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:
 for m in archive.getmembers():
  p=(project/m.name).resolve()
  if not p.is_relative_to(project.resolve()) or not m.isfile():raise ValueError('Invalid payload')
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.extractfile(m).read())
def release():
 from google.colab import runtime
 runtime.unassign()
def deadline():
 time.sleep(SECONDS_LEFT);release()
threading.Thread(target=deadline,daemon=True).start()
def snapshot():
 with lock:
  tmp=Path('/content/copollm-snapshot.tmp')
  with zipfile.ZipFile(tmp,'w') as z:
   for p in out.rglob('*'):
    if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):
     data=p.read_bytes()
     if p.suffix=='.jsonl':data=data[:data.rfind(b'\n')+1]
     z.writestr(str(p.relative_to(out)),data)
  tmp.replace('/content/copollm-snapshot.zip')
def heartbeat():
 while not finished.wait(15):
  try:
   snapshot();steps=sum(1 for _ in (out/'training.jsonl').open()) if (out/'training.jsonl').exists() else 0
   print('TRAINING PROGRESS',round(time.monotonic()-started),'steps',steps,flush=True)
  except Exception as e:print('Snapshot warning',repr(e),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
(out/'provenance.json').write_text(json.dumps({'code_commit':CODE_COMMIT,'payload_sha256':PAYLOAD_SHA256,'scope':'20-step pipeline controls, no API, no test tuning'},indent=2))
snapshot()
process=None
try:
 with (out/'setup.log').open('a') as log:
  subprocess.run([sys.executable,'-m','pip','install','-r',str(project/'requirements.txt')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,SECONDS_LEFT-180))
 command=[sys.executable,'-u',str(project/'smoke_train.py'),'--train-data',str(project/'train.json'),'--source',str(project/'sft_unified.py'),'--out',str(out),'--steps','20']
 with (out/'job.log').open('a',buffering=1) as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while process.poll() is None:
   if time.monotonic()-started>SECONDS_LEFT-180:raise TimeoutError('Training stopped with artifact-transfer reserve')
   time.sleep(2)
 if process.returncode:raise RuntimeError('Training subprocess failed; inspect job.log')
 artifact=Path('/content/training-artifacts');artifact.mkdir(exist_ok=True)
 bundle=artifact/'training-results.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_STORED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p.suffix!='.pt':z.write(p,str(p.relative_to(out)))
 parts=[]
 with bundle.open('rb') as source:
  index=0
  while data:=source.read(16*1024*1024):
   name=f'artifact-{index:03d}.part';(artifact/name).write_bytes(data)
   parts.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
 (out/'artifact_parts.json').write_text(json.dumps({'parts':parts,'archive_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()},indent=2))
 print('TRAINING COMPLETE', (out/'summary.json').read_text(),flush=True)
finally:
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
 snapshot();finished.set()
 # Independent cleanup if the local controller disappears during artifact retrieval.
 timer=threading.Timer(max(1,SECONDS_LEFT-(time.monotonic()-started)),release);timer.daemon=True;timer.start()
