"""Allocate A100 for two 20-step training controls; bound time/CU and verify release."""
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
    p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--units',type=float,default=5)
    p.add_argument('--high-mem',action='store_true')
    p.add_argument('--gpus',nargs='+',choices=['A100','T4'],default=['A100'])
    a=p.parse_args()
    if not 3<a.minutes<=30 or not 1<a.units<=5:
        p.error('This authorized round supports at most 30 minutes and 5 CU')
    round_id='train-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
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
        'minutes_cap':a.minutes,'session':session,'scope':'two real-data 20-step SFT controls, no external API'},indent=2))
    emit(f'STATE_DIR {state}; starting balance {balance}')
    started=time.monotonic()
    last_usage=0
    last_rate=rate
    last_snapshot=0
    last_good_snapshot=[time.monotonic()]
    def fetch_snapshot():
        temp=state/'snapshot.download.zip'
        r=cli('download','-s',session,'/content/copollm-snapshot.zip',str(temp),timeout=60)
        if r.returncode: return False
        with zipfile.ZipFile(temp) as z:
            for name in z.namelist():
                target=(state/'results'/name).resolve()
                if not target.is_relative_to((state/'results').resolve()):
                    raise ValueError('unsafe snapshot path')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(name))
        temp.replace(state/'snapshot.zip')
        last_good_snapshot[0]=time.monotonic()
        emit('SNAPSHOT saved locally')
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
            reserve=max(.4,last_rate/60)
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
                        try:
                            if not fetch_snapshot(): emit('Snapshot not ready yet')
                        except Exception as exc:
                            emit('Snapshot retryable failure: '+repr(exc))
                            if time.monotonic()-last_good_snapshot[0]>240:
                                raise RuntimeError('No persistent snapshot for 4 minutes')
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
            rc=monitored(['new','-s',session,'--gpu',gpu]+(['--high-mem'] if a.high_mem else []),'allocate-'+gpu,timeout=300)
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
            payload_paths={'smoke_train.py':PROJECT/'reproduction/training/smoke_train.py',
                'data.py':PROJECT/'reproduction/training/data.py','requirements.txt':HERE/'requirements-colab.txt',
                'train.json':Path('/Users/yiwenmu/Documents/ChatGPT/cbt/copollm-source/data/train.json'),
                'sft_unified.py':Path('/Users/yiwenmu/Documents/ChatGPT/cbt/copollm-source/task4/sft_unified.py')}
            for name,path in payload_paths.items():
                info=tarfile.TarInfo(name);body=path.read_bytes();info.size=len(body)
                archive.addfile(info,io.BytesIO(body))
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
        payload=data.getvalue()
        prelude='\n'.join([f'PAYLOAD={base64.b64encode(payload).decode()!r}',
            f'PAYLOAD_SHA256={hashlib.sha256(payload).hexdigest()!r}',
            f'CODE_COMMIT={commit!r}',f'ROUND_ID={round_id!r}',
            f'SECONDS_LEFT={max(1,a.minutes*60-(time.monotonic()-started)-45)!r}'])+'\n'
        driver=state/'remote_driver.py'
        driver.write_text(prelude+(PROJECT/'reproduction/training/remote_train.py').read_text())
        rc=monitored(['exec','-s',session,'--timeout','180','-f',str(driver)],'execute')
        if not fetch_snapshot(): raise RuntimeError('No final snapshot retrieved')
        emit('EXECUTE returncode '+str(rc))
        emit((state/'execute.log').read_text()[-6500:])
        summary=json.loads((state/'results/summary.json').read_text())
        if set(summary['arms'])!={'generation_only','joint'} or any(v['optimizer_steps']!=20 for v in summary['arms'].values()):
            raise RuntimeError('Both 20-step training checks did not complete')
        from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
        parts=json.loads((state/'results/artifact_parts.json').read_text())
        partdir=state/'adapter-parts';partdir.mkdir(exist_ok=True)
        def download_part(part):
            target=partdir/part['name']
            for retry in range(3):
                result=cli('download','-s',session,'/content/training-artifacts/'+part['name'],str(target),timeout=60)
                if result.returncode==0 and hashlib.sha256(target.read_bytes()).hexdigest()==part['sha256']:return target
            raise RuntimeError('Artifact download failed: '+part['name'])
        pool=ThreadPoolExecutor(max_workers=2)
        try:
            pending={pool.submit(download_part,part) for part in parts['parts']}
            while pending:
                done,pending=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for item in done:emit('ARTIFACT '+str(item.result().name))
                check_budget()
        finally:pool.shutdown(wait=False,cancel_futures=True)
        bundle=state/'training-results.zip'
        with bundle.open('wb') as dest:
            for part in parts['parts']:dest.write((partdir/part['name']).read_bytes())
        if hashlib.sha256(bundle.read_bytes()).hexdigest()!=parts['archive_sha256']:raise RuntimeError('Artifact SHA mismatch')
        with zipfile.ZipFile(bundle) as z:
            if z.testzip() is not None:raise RuntimeError('Artifact CRC failed')
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
