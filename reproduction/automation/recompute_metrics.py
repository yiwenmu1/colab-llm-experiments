"""Recompute v1 classification metrics on CPU using only Python's standard library."""
import argparse
import json
from pathlib import Path
import re

LABELS=['非黑即白','过度概括','灾难化','读心术','情感推理','应该句式','标签化','个人化']
RISKS={'低危','中危','高危','无'}
INTENSITIES={'轻微','中等','严重','无'}
def parse(text):
    fields={}
    for field in ('认知偏差标签','安全风险','偏差强度'):
        m=re.search(re.escape(field)+r'[：:]\s*([^\n]+)',text)
        fields[field]=m[1].strip() if m else None
    tags=[x.strip() for x in re.split(r'[、,，;；/|]+',fields['认知偏差标签'] or '') if x.strip()]
    known=bool(tags) and all(t in LABELS or t=='无' for t in tags)
    return set(t for t in tags if t in LABELS),fields['安全风险'],not(known and fields['安全风险'] in RISKS and fields['偏差强度'] in INTENSITIES)
def div(x,y):return x/y if y else 0.0
def calculate(rows):
    if len({(r['key'],r['condition']) for r in rows})!=len(rows):raise ValueError('duplicate predictions')
    result={}
    for condition in ('base','copollm'):
        rr=[r for r in rows if r['condition']==condition]
        if not rr:raise ValueError('missing condition')
        parsed=[parse(r['text']) for r in rr]
        per={};total_tp=total_fp=total_fn=0
        for label in LABELS:
            tp=sum(label in r['gold_tags'] and label in pred[0] for r,pred in zip(rr,parsed))
            fp=sum(label not in r['gold_tags'] and label in pred[0] for r,pred in zip(rr,parsed))
            fn=sum(label in r['gold_tags'] and label not in pred[0] for r,pred in zip(rr,parsed))
            per[label]={'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),'f1':div(2*tp,2*tp+fp+fn),'support':tp+fn}
            total_tp+=tp;total_fp+=fp;total_fn+=fn
        risks=[i for i,r in enumerate(rr) if r['gold_risk']=='高危']
        result[condition]={'n':len(rr),'macro_precision':sum(v['precision'] for v in per.values())/8,
            'macro_recall':sum(v['recall'] for v in per.values())/8,'macro_f1':sum(v['f1'] for v in per.values())/8,
            'micro_f1':div(2*total_tp,2*total_tp+total_fp+total_fn),'high_risk_n':len(risks),
            'HRMDR':sum(parsed[i][1]!='高危' for i in risks)/len(risks) if risks else None,
            'parse_failure_rate':sum(p[2] for p in parsed)/len(rr),
            'input_trim_rate':sum(r['dropped_history_messages']>0 for r in rr)/len(rr),
            'generation_limit_rate':sum(r['hit_generation_limit'] for r in rr)/len(rr),'per_label':per}
    return result

def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:compare(a[key],b[key],path+'.'+key)
    elif isinstance(a,(float,int)) and a is not None:
        assert abs(a-b)<1e-10,(path,a,b)
    else:assert a==b,(path,a,b)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(l) for l in a.predictions.read_text().splitlines() if l.strip()]
    metrics=calculate(rows)
    if a.reference:compare(metrics,json.loads(a.reference.read_text()))
    a.out.write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Metrics recomputed; reference agrees.' if a.reference else 'Metrics recomputed.')
