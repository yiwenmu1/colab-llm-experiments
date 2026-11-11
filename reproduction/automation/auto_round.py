"""Allocate A100/T4, run frozen v1 full evaluation, bound time/CU, verify release."""
import argparse
import base64
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import zipfile
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent

def parse_usage(text):
    b=re.search(r'Current balance: ([\d.]+)',text)
    r=re.search(r'Usage rate: ([\d.]+)',text)
    if not b or not r:
        raise RuntimeError('Unable to read compute-unit balance/rate')
    return float(b[1]),float(r[1])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--units',type=float,default=25)
    p.add_argument('--gpus',nargs='+',choices=['A100','T4'],default=['A100','T4'])
    a=p.parse_args()
    if not 3<a.minutes<=120 or not 1<a.units<=25:
        p.error('This authorized round supports at most 120 minutes and 25 CU')
    round_id='eval-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    state=PROJECT/'runs'/round_id
    state.mkdir(parents=True)
    session=round_id
    def emit(s):
        print(s,flush=True)
        with (state/'controller.log').open('a') as f: f.write(s+'\n')
    def cli(*args,timeout=45):
        return subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout)
    before=cli('usage')
    if before.returncode: raise RuntimeError(before.stdout+before.stderr)
    balance,rate=parse_usage(before.stdout)
    if balance<a.units: raise RuntimeError('Balance below authorized budget; stop before allocation')
    if cli('sessions').returncode: raise RuntimeError('Unable to inspect existing sessions')
    (state/'budget.json').write_text(json.dumps({'initial_balance':balance,'unit_cap':a.units,
        'minutes_cap':a.minutes,'session':session,'scope':'981 x 2 classify, frozen v1, no API'},indent=2))
    emit(f'STATE_DIR {state}; starting balance {balance}')
    seed={}
    seed_dir=state/'seed';seed_dir.mkdir()
    for name in ('qwen3_8b_nf4_ctx2048_v1_diagnosis.jsonl','qwen3_8b_nf4_ctx2048_v1_manifest.json',
                 'qwen3_8b_nf4_ctx2048_v1_revisions.json','sample_manifest.json',
                 'qwen3_8b_nf4_ctx2048_v1_metrics.json','qwen3_8b_nf4_ctx2048_v1_pip-freeze.txt'):
        path=seed_dir/name
        r=cli('download','-s','llm','/content/drive/MyDrive/colab-llm-experiments/public_subset_eval/'+name,str(path))
        if r.returncode: raise RuntimeError('CPU Drive seed download failed: '+name)
        seed[name]=base64.b64encode(path.read_bytes()).decode()
    started=time.monotonic()
    last_usage=0
    last_rate=rate
    last_snapshot=0
    def fetch_snapshot():
        temp=state/'snapshot.download.zip'
        r=cli('download','-s',session,'/content/copollm-snapshot.zip',str(temp),timeout=30)
        if r.returncode: return False
        with zipfile.ZipFile(temp) as z:
            for name in z.namelist():
                target=(state/'results'/name).resolve()
                if not target.is_relative_to((state/'results').resolve()):
                    raise ValueError('unsafe snapshot path')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(name))
        temp.replace(state/'snapshot.zip')
        # Existing CPU session already has Drive authorization; upload atomic snapshot only.
        r=cli('upload','-s','llm',str(state/'snapshot.zip'),
            '/content/drive/MyDrive/colab-llm-experiments/'+round_id+'-snapshot.zip',timeout=30)
        if r.returncode: raise RuntimeError('Persistent snapshot upload failed')
        emit('SNAPSHOT saved locally and to Drive')
        return True
    def check_budget():
        nonlocal last_usage,last_rate
        now=time.monotonic()
        if now-started>=a.minutes*60-90:
            raise RuntimeError('Wall-time budget reached; releasing with cleanup reserve')
        if now-last_usage>=30:
            u=cli('usage',timeout=20)
            if u.returncode: raise RuntimeError('Usage monitoring failed; stop safely')
            current,last_rate=parse_usage(u.stdout)
            last_usage=time.monotonic()
            spent=max(0,balance-current)
            emit(f'BUDGET spent={spent:.2f} rate={last_rate:.2f}/h elapsed={(now-started)/60:.1f}m')
            # Account-level balance includes other sessions; conservative accounting.
            reserve=max(1,last_rate*2/60)
            if spent>=a.units-reserve:
                raise RuntimeError('Compute-unit budget reached; release with reserve')
    def monitored(args,label,timeout=None):
        with (state/(label+'.log')).open('w') as log:
            proc=subprocess.Popen(['colab',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            begin=time.monotonic()
            last_snapshot=time.monotonic()
            try:
                while proc.poll() is None:
                    check_budget()
                    if label=='execute' and time.monotonic()-last_snapshot>=45:
                        if not fetch_snapshot(): emit('Snapshot not ready yet')
                        last_snapshot=time.monotonic()
                    if timeout and time.monotonic()-begin>timeout:
                        raise RuntimeError(label+' timed out')
                    time.sleep(2)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            return proc.returncode
    exitcode=1
    allocated=False
    endpoint=None
    def interrupt(signum, frame):
        raise InterruptedError('Controller interrupted; releasing owned session')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for gpu in a.gpus:
            emit('REQUEST '+gpu)
            rc=monitored(['new','-s',session,'--gpu',gpu],'allocate-'+gpu,timeout=300)
            emit((state/('allocate-'+gpu+'.log')).read_text())
            if rc==0:
                allocated=True
                listing=cli('sessions').stdout
                line=next((line for line in listing.splitlines() if f'[{session}]' in line), '')
                endpoint=line.split('] ',1)[-1].split(' |',1)[0] if line else None
                if not endpoint: raise RuntimeError('Cannot identify allocated endpoint')
                emit('ALLOCATED '+line)
                break
            # Do not leak partial allocations before trying fallback.
            cli('stop','-s',session)
        if not allocated: raise RuntimeError('A100 and T4 unavailable; no retries')
        check_budget()
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w:gz') as archive:
            for name in ('evaluate.py','job.py','requirements-colab.txt'):
                path=HERE/name
                info=tarfile.TarInfo('reproduction/automation/'+name)
                body=path.read_bytes();info.size=len(body)
                archive.addfile(info,io.BytesIO(body))
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
        payload=data.getvalue()
        prelude='\n'.join([f'SEED={seed!r}',f'PAYLOAD={base64.b64encode(payload).decode()!r}',
            f'PAYLOAD_SHA256={hashlib.sha256(payload).hexdigest()!r}',
            f'CODE_COMMIT={commit!r}',f'ROUND_ID={round_id!r}',
            f'SECONDS_LEFT={max(1,a.minutes*60-(time.monotonic()-started)-45)!r}'])+'\n'
        driver=state/'remote_driver.py'
        driver.write_text(prelude+(HERE/'remote_round.py').read_text())
        rc=monitored(['exec','-s',session,'--timeout','180','-f',str(driver)],'execute')
        if not fetch_snapshot(): raise RuntimeError('No final snapshot retrieved')
        emit('EXECUTE returncode '+str(rc))
        emit((state/'execute.log').read_text()[-6500:])
        exitcode=rc
    except BaseException as e:
        emit('ROUND stopped: '+repr(e))
    finally:
        if allocated:
            try: fetch_snapshot()
            except Exception as e: emit('FINAL SNAPSHOT error '+repr(e))
        released=False
        for attempt in range(3):
            try:
                stop=cli('stop','-s',session,timeout=30)
                emit('RELEASE '+stop.stdout+stop.stderr)
                sessions=cli('sessions',timeout=20)
                if sessions.returncode==0 and f'[{session}]' not in sessions.stdout and (not endpoint or endpoint not in sessions.stdout):
                    released=True;break
            except Exception as e:
                emit('RELEASE error '+repr(e))
        final={'session':session,'release_verified':released,'returncode':exitcode,
               'elapsed_minutes':round((time.monotonic()-started)/60,2)}
        try:
            u=cli('usage'); current,_=parse_usage(u.stdout)
            final.update(final_balance=current,account_units_spent=round(balance-current,2))
        except Exception as e: final['usage_error']=repr(e)
        (state/'final.json').write_text(json.dumps(final,indent=2))
        emit(json.dumps(final))
        if not released: exitcode=1
    return exitcode

if __name__=='__main__':
    raise SystemExit(main())
