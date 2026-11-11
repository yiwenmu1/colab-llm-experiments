"""SSH entry point exported from the starter Notebook. Mount Drive in Colab first.
Default reproduces the existing 10-case protocol; full evaluation requires explicit opt-in.
"""
import argparse
import os
from pathlib import Path
import subprocess, json, sys, platform, time, hashlib, random, ast, re
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support
from contextlib import nullcontext


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--run-name', default='qwen3_8b_nf4_ctx2048_v1')
p.add_argument('--allow-ephemeral-output', action='store_true', help='Controller snapshots results to persistent storage')
p.add_argument('--limit', type=int, default=10)
p.add_argument('--full', action='store_true', help='Only after reviewing output protocol on validation data')
a = p.parse_args()
if a.limit <= 0 or not re.fullmatch(r'[A-Za-z0-9_.-]+', a.run_name):
    p.error('limit 必须为正数，run-name 只能含字母数字、点、横线和下划线')
OUT = a.out
if not a.allow_ephemeral_output and not Path('/content/drive/MyDrive').is_dir():
    raise RuntimeError('请先在 Colab Notebook 中挂载 Drive，再从 SSH 启动')
OUT.mkdir(parents=True, exist_ok=True)
RUN_NAME = a.run_name
LIMIT = None if a.full else a.limit
MAX_INPUT = 2048
MAX_NEW = 128
assert torch.cuda.is_available(), '需要 GPU 运行时'
random.seed(42); np.random.seed(42); torch.manual_seed(42)

results_path = OUT/(RUN_NAME+'_diagnosis.jsonl')
seen = set()
if results_path.exists():
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row['key'], row['condition'])
        if key in seen:
            raise RuntimeError(f'重复预测键 {key}，请先备份核查')
        seen.add(key)
    if seen and not (OUT/(RUN_NAME+'_manifest.json')).exists():
        raise RuntimeError('已有预测但缺少配置清单，拒绝混用')
REPO = Path('/content/CoPoLLM-for-ACL-2026')
COMMIT = 'cfe92bb0228da702af910adf31b1a64a5e13cb8b'
if not REPO.exists():
    subprocess.run(['git','clone','https://github.com/Chips98/CoPoLLM-for-ACL-2026.git',str(REPO)], check=True)
subprocess.run(['git','checkout',COMMIT],cwd=REPO,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == COMMIT
train = json.loads((REPO/'data/train.json').read_text())
test = json.loads((REPO/'data/test.json').read_text())
print('train dialogues',len(train),'test dialogues',len(test))
# 只解析源码中的常量，不 import 训练脚本，不执行论文或数据中的指令。
needed = {'CLS_TASK_ID','GEN_TASK_ID','CLS_MARKER','GEN_MARKER',
          'RESPONSE_TEMPLATE','SYS_PROMPT_CLASSIFY','SYS_PROMPT_GENERATION'}
constants = {}
for node in ast.parse((REPO/'task4/sft_unified.py').read_text()).body:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in needed:
                constants[target.id] = ast.literal_eval(node.value)
assert set(constants) == needed
print('已读取源码任务模板')
LABELS = ['非黑即白','过度概括','灾难化','读心术','情感推理','应该句式','标签化','个人化']
VALID_RISKS = {'低危','中危','高危','无'}
VALID_INTENSITIES = {'轻微','中等','严重','无'}
def normalize_tags(value):
    if isinstance(value,str):
        return [x.strip() for x in re.split(r'[、,，;；/|]+',value) if x.strip()]
    if isinstance(value,list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []

def make_prompt(task, history, user):
    cls = task == 'classify'
    task_id = constants['CLS_TASK_ID' if cls else 'GEN_TASK_ID']
    marker = constants['CLS_MARKER' if cls else 'GEN_MARKER']
    system = constants['SYS_PROMPT_CLASSIFY' if cls else 'SYS_PROMPT_GENERATION']
    history_text = ''.join(('患者：' if m['role']=='user' else '治疗师：')+m['content']+'\n' for m in history)
    return f"{task_id}\n{marker}\n{system}\n历史对话：\n{history_text}患者：{user}" + constants['RESPONSE_TEMPLATE']

samples=[]
excluded=0
for di,conv in enumerate(test):
    history=[]
    for ti,msg in enumerate(conv['messages']):
        if msg.get('role')=='user' and isinstance(msg.get('annotation'),dict):
            a=msg['annotation']; tags=normalize_tags(a.get('bias_tags'))
            valid = bool(tags) and all(x in LABELS for x in tags) and a.get('risk_level') in VALID_RISKS and a.get('bias_intensity') in VALID_INTENSITIES
            if valid:
                samples.append(dict(key=f'{di}:{ti}',history=list(history),user=msg['content'],
                                    gold_tags=tags,gold_risk=a['risk_level'],gold_intensity=a['bias_intensity']))
            else:
                excluded+=1
        if msg.get('role') in ('user','assistant'):
            history.append({'role':msg['role'],'content':msg['content']})
print('有效诊断样本',len(samples),'排除无效标注',excluded,
      '高危',sum(x['gold_risk']=='高危' for x in samples))
assert len(samples)==981 and sum(x['gold_risk']=='高危' for x in samples)==42
(OUT/'sample_manifest.json').write_text(json.dumps([{k:v for k,v in s.items() if k not in ('history','user')} for s in samples],ensure_ascii=False,indent=2))
BASE = 'Qwen/Qwen3-8B'
ADAPTER_REPO = 'Chips95/Lora_Adapter_for_ACL2026_CoPoLLM'
SUBDIR = 'lora_adapter_for_qwen3-8b'
api = HfApi()
# 首次解析后写入锁文件；后续续跑沿用相同 revisions。
lock_path=OUT/(RUN_NAME+'_revisions.json')
if lock_path.exists():
    lock=json.loads(lock_path.read_text())
else:
    lock={'base':api.model_info(BASE).sha,'adapter':api.model_info(ADAPTER_REPO).sha}
    lock_path.write_text(json.dumps(lock,indent=2))
local = Path(snapshot_download(ADAPTER_REPO,revision=lock['adapter'],allow_patterns=[
    SUBDIR+'/adapter_config.json',SUBDIR+'/adapter_model.safetensors',
    SUBDIR+'/tokenizer*',SUBDIR+'/special_tokens_map.json',SUBDIR+'/added_tokens.json',
    SUBDIR+'/vocab.json',SUBDIR+'/merges.txt']))/SUBDIR
config=json.loads((local/'adapter_config.json').read_text())
print('作者基座路径',config['base_model_name_or_path'],'实际使用',BASE)
tokenizer=AutoTokenizer.from_pretrained(local)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token=tokenizer.eos_token
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
base=AutoModelForCausalLM.from_pretrained(BASE,revision=lock['base'],
    quantization_config=quant,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
assert len(tokenizer)<=base.get_input_embeddings().num_embeddings, '词表不匹配，先核查 tokenizer，勿盲目 resize'
model=PeftModel.from_pretrained(base,str(local))
model.eval()
(OUT/(RUN_NAME+'_pip-freeze.txt')).write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
manifest={'commit':COMMIT,'base':BASE,'adapter_repo':ADAPTER_REPO,'subdir':SUBDIR,'revisions':lock,
          'max_input':MAX_INPUT,'max_new':MAX_NEW,'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(0),'quantization':'NF4 double quant FP16',
          'template_sha256':hashlib.sha256(json.dumps(constants,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
manifest_path=OUT/(RUN_NAME+'_manifest.json')
if manifest_path.exists():
    old=json.loads(manifest_path.read_text())
    for k in ('commit','base','adapter_repo','subdir','revisions','max_input','max_new','quantization','template_sha256'):
        assert old[k]==manifest[k], f'实验条件改变：{k}，请换 RUN_NAME'
else:
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
def fit_prompt(sample,task):
    history=list(sample['history']); dropped=0
    while True:
        prompt=make_prompt(task,history,sample['user'])
        inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        if inputs['input_ids'].shape[1]<=MAX_INPUT:
            return inputs,dropped
        if not history:
            raise ValueError('当前发言+任务指令已超长；提高 MAX_INPUT 并更换 RUN_NAME')
        # 按轮次删除最早历史，保留任务指令、当前发言和响应锚点。
        n=1
        if history[0]['role']=='user' and len(history)>1 and history[1]['role']=='assistant':
            n=2
        history=history[n:]; dropped+=n

@torch.inference_mode()
def predict(sample, task='classify', use_adapter=True):
    inputs,dropped=fit_prompt(sample,task)
    inputs={k:v.to('cuda:0') for k,v in inputs.items()}
    n=inputs['input_ids'].shape[1]
    ctx=nullcontext() if use_adapter else model.disable_adapter()
    with ctx:
        out=model.generate(**inputs,do_sample=False,max_new_tokens=MAX_NEW,
                           pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(out[0,n:],skip_special_tokens=True)
    return {'text':text,'input_tokens':n,'dropped_history_messages':dropped,
            'hit_generation_limit':out.shape[1]-n>=MAX_NEW}


selected=samples if LIMIT is None else samples[:LIMIT]
results_path=OUT/(RUN_NAME+'_diagnosis.jsonl')
existing=[]
if results_path.exists():
    for line in results_path.read_text().splitlines():
        if line.strip(): existing.append(json.loads(line))
done={(x['key'],x['condition']) for x in existing}
started=time.time(); count=0
with results_path.open('a',encoding='utf-8') as f:
    for s in selected:
        for condition,use_adapter in [('base',False),('copollm',True)]:
            if (s['key'],condition) in done: continue
            prediction=predict(s,use_adapter=use_adapter)
            row={k:s[k] for k in ('key','gold_tags','gold_risk','gold_intensity')}
            row.update(condition=condition,**prediction)
            f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
            done.add((s['key'],condition)); count+=1
            if count%10==0: print('本次完成',count,'预测，秒数',round(time.time()-started,1))
print('结果',results_path)
def parse_prediction(text):
    fields={}
    for field in ('认知偏差标签','安全风险','偏差强度'):
        m=re.search(re.escape(field)+r'[：:]\s*([^\n]+)',text)
        fields[field]=m.group(1).strip() if m else None
    tags=normalize_tags(fields['认知偏差标签'])
    known=bool(tags) and all(x in LABELS or x=='无' for x in tags)
    pred_tags=[x for x in tags if x in LABELS]
    risk=fields['安全风险']; intensity=fields['偏差强度']
    return pred_tags,risk,intensity,not (known and risk in VALID_RISKS and intensity in VALID_INTENSITIES)

wanted={s['key'] for s in selected}
rows=[json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
report={}
for condition in ('base','copollm'):
    current=[x for x in rows if x['condition']==condition and x['key'] in wanted]
    assert len(current)==len(selected), '预测未完成，请继续运行前一格'
    parsed=[parse_prediction(x['text']) for x in current]
    gold=np.array([[int(t in x['gold_tags']) for t in LABELS] for x in current])
    pred=np.array([[int(t in p[0]) for t in LABELS] for p in parsed])
    macro=precision_recall_fscore_support(gold,pred,average='macro',zero_division=0)
    micro=precision_recall_fscore_support(gold,pred,average='micro',zero_division=0)
    risks=[i for i,x in enumerate(current) if x['gold_risk']=='高危']
    missed=sum(parsed[i][1]!='高危' for i in risks)
    per=precision_recall_fscore_support(gold,pred,average=None,zero_division=0)
    report[condition]={'n':len(current),'macro_precision':float(macro[0]),'macro_recall':float(macro[1]),
        'macro_f1':float(macro[2]),'micro_f1':float(micro[2]),'high_risk_n':len(risks),
        'HRMDR':missed/len(risks) if risks else None,
        'parse_failure_rate':sum(p[3] for p in parsed)/len(parsed),
        'input_trim_rate':sum(x['dropped_history_messages']>0 for x in current)/len(current),
        'generation_limit_rate':sum(x['hit_generation_limit'] for x in current)/len(current),
        'per_label':{t:{'precision':float(per[0][i]),'recall':float(per[1][i]),'f1':float(per[2][i]),'support':int(per[3][i])} for i,t in enumerate(LABELS)}}
print(json.dumps(report,ensure_ascii=False,indent=2))
(OUT/(RUN_NAME+'_metrics.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))