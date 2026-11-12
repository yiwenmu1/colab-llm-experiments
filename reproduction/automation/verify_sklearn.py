"""Independent sklearn verification of saved v1 predictions; CPU only."""
import argparse,json,re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from recompute_metrics import LABELS,compare
p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('reference',type=Path)
a=p.parse_args();rows=[json.loads(l) for l in a.predictions.read_text().splitlines() if l.strip()]
reference=json.loads(a.reference.read_text())
for condition in ('base','copollm'):
    rr=[r for r in rows if r['condition']==condition]
    predicted=[]
    for r in rr:
        match=re.search(r'认知偏差标签[：:]\s*([^\n]+)',r['text'])
        tags=[x.strip() for x in re.split(r'[、,，;；/|]+',match[1].strip()) if x.strip()] if match else []
        predicted.append([int(label in tags) for label in LABELS])
    gold=np.array([[int(label in r['gold_tags']) for label in LABELS] for r in rr]);pred=np.array(predicted)
    precision,recall,f1,_=precision_recall_fscore_support(gold,pred,average='macro',zero_division=0)
    micro=precision_recall_fscore_support(gold,pred,average='micro',zero_division=0)[2]
    per=precision_recall_fscore_support(gold,pred,average=None,zero_division=0)
    for key,value in [('macro_precision',precision),('macro_recall',recall),('macro_f1',f1),('micro_f1',micro)]:
        assert abs(value-reference[condition][key])<1e-10,(condition,key,value,reference[condition][key])
    for i,label in enumerate(LABELS):
        compare({'precision':float(per[0][i]),'recall':float(per[1][i]),'f1':float(per[2][i]),'support':int(per[3][i])},reference[condition]['per_label'][label])
print('Independent sklearn verification passed for both conditions.')
