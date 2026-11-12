"""Recompute small validation diagnostics without exporting dialogue or predictions."""
import argparse,collections,json,re
from pathlib import Path

def fields(text):
    result=[]
    for name in ('认知偏差标签','安全风险','偏差强度'):
        match=re.search(name+r'[：:]([^\n]+)',text)
        result.append(match.group(1).strip() if match else None)
    return result

def analyze(directory):
    result={'scope':'8 fixed validation examples only; no final-test efficacy claim','conditions':{}}
    for name in ('initial','generation_only','joint'):
        rows=json.loads((directory/(name+'_validation_predictions.json')).read_text())
        predictions=[fields(row['text']) for row in rows];targets=[fields(row['target']) for row in rows]
        result['conditions'][name]={'count':len(rows),'all_fields_found':sum(all(v is not None for v in p) for p in predictions),
            'hit_limit':sum(row['hit_limit'] for row in rows),
            'exact_field_matches':{key:sum(p[i] is not None and p[i]==t[i] for p,t in zip(predictions,targets)) for i,key in enumerate(('label_text','risk','intensity'))},
            'predicted_label_counts':dict(collections.Counter(p[0] for p in predictions))}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.results),ensure_ascii=False,indent=2))
