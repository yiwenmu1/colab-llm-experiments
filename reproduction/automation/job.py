"""Bound one command by time; persist status/logs. Release runtime in notebook, not SSH."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--timeout-minutes', type=float, default=120)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.timeout_minutes <= 0:
        p.error('需要命令和正数时间预算')
    a.state_dir.mkdir(parents=True, exist_ok=True)
    # Linux flock prevents accidental concurrent jobs in the same state directory.
    lock = (a.state_dir / '.job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    status = {'supervisor_pid': os.getpid(), 'state': 'starting', 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'command': command, 'timeout_minutes': a.timeout_minutes}
    def save(**updates):
        status.update(updates, elapsed_seconds=round(time.time()-started, 1))
        tmp = a.state_dir / 'status.tmp'
        with tmp.open('w') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(a.state_dir / 'status.json')
    stop = [False]
    def handle_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    child = None
    save()
    try:
        with (a.state_dir / 'job.log').open('a', buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            save(state='running', pid=child.pid)
            while child.poll() is None:
                if stop[0] or time.time()-started >= a.timeout_minutes*60:
                    reason = 'stopped' if stop[0] else 'timed_out'
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    save(state=reason, returncode=child.returncode)
                    return 124 if reason == 'timed_out' else 130
                save(state='running')
                time.sleep(2)
            save(state='completed' if child.returncode == 0 else 'failed', returncode=child.returncode)
            return child.returncode
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        save(state='failed', error=repr(exc))
        raise

if __name__ == '__main__':
    raise SystemExit(main())
