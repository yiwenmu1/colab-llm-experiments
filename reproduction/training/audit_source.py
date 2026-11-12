"""Reproduce data-construction bugs in the original source without importing training libraries."""
import argparse,ast,json,random,io,contextlib
from pathlib import Path
from data import build_examples,split_dialogues,templates
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
source=a.source_root/'task4/sft_unified.py'
class Dataset:
    @staticmethod
    def from_list(rows):return rows
ns={'json':json,'random':random,'Dataset':Dataset,**templates(source)}
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_mixed_dataset')
node.returns=None
for arg in node.args.args:arg.annotation=None
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
with contextlib.redirect_stdout(io.StringIO()):original=ns['build_mixed_dataset'](str(a.source_root/'data/train.json'),None,2.0)
gen=[r for r in original if r['task_type']=='generation']
duplicate=sum(r['messages'][-2]['role']=='user' and r['messages'][-3]['role']=='user' and r['messages'][-2]['content']==r['messages'][-3]['content'] for r in gen)
raw=json.loads((a.source_root/'data/train.json').read_text());parts,manifest=split_dialogues(raw)
result={'upstream_generation_samples':len(gen),'upstream_duplicate_current_user':duplicate,'fixed':{},
        'split_dialogues':{k:len(v) for k,v in parts.items()},'test_data_used_for_training':False}
for part,entries in parts.items():
    examples,excluded=build_examples(entries)
    result['fixed'][part]={'classification':sum(r['task']=='classify' for r in examples),'generation':sum(r['task']=='generation' for r in examples),'excluded_annotation_or_empty':len(excluded)}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
