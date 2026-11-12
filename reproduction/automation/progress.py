"""Summarize complete saved JSONL rows across disjoint inference shards."""
import argparse,json,re
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('round_dir',type=Path);a=p.parse_args()
root=a.round_dir/'results'
rows={}
for f in (root/'public_subset_eval').glob('*_diagnosis.jsonl'):
    for line in f.read_text().splitlines():
        if line.strip():
            r=json.loads(line);rows[(r['key'],r['condition'])]=r
rates=[];worker_progress={}
for f in (root/'jobs').rglob('job.log'):
    found=re.findall(r'本次完成 (\d+) 预测，秒数 ([\d.]+)',f.read_text())
    if found:
        n,t=map(float,found[-1]);worker_progress[f.parent.name]={'new':int(n),'seconds':t}
        if len(found)>1:
            n0,t0=map(float,found[-2]);rates.append((n-n0)/(t-t0))
        elif t:rates.append(n/t)
summary={'unique_predictions':len(rows),'target':1962,'conditions':dict(Counter(r['condition'] for r in rows.values())),
         'workers':worker_progress}
if len(rates)==4 and sum(rates)>0:
    summary.update(recent_predictions_per_minute=round(sum(rates)*60,1),estimated_remaining_minutes=round((1962-len(rows))/sum(rates)/60,1))
print(json.dumps(summary,ensure_ascii=False))
