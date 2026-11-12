"""Leakage-aware real-data preparation and strict response-only tokenization."""
import ast,hashlib,json,random,re
from pathlib import Path
LABELS=['非黑即白','过度概括','灾难化','读心术','情感推理','应该句式','标签化','个人化']
RISKS={'低危','中危','高危','无'};INTENSITIES={'轻微','中等','严重','无'}
def normalize_tags(value):
    if isinstance(value,list):return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value,str):return [t.strip() for t in re.split(r'[、,，;；/|]+',value) if t.strip()]
    return []
def annotation(value):
    if not isinstance(value,dict):return None
    tags=normalize_tags(value.get('bias_tags'))
    if not tags or any(t not in LABELS for t in tags):return None
    if value.get('risk_level') not in RISKS or value.get('bias_intensity') not in INTENSITIES:return None
    return dict(tags=tags,risk=value['risk_level'],intensity=value['bias_intensity'])
def templates(source):
    names={'CLS_TASK_ID','GEN_TASK_ID','CLS_MARKER','GEN_MARKER','RESPONSE_TEMPLATE','SYS_PROMPT_CLASSIFY','SYS_PROMPT_GENERATION'}
    result={}
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:result[target.id]=ast.literal_eval(node.value)
    if set(result)!=names:raise ValueError('Missing author templates')
    return result

def split_dialogues(dialogues,seed=42,val_fraction=.1):
    # Duplicate conversations stay in one partition, including duplicates with different IDs.
    groups={}
    for index,dialogue in enumerate(dialogues):
        content=[{'role':m.get('role'),'content':m.get('content','')} for m in dialogue.get('messages',[]) if m.get('role') in ('user','assistant')]
        fingerprint=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        groups.setdefault(fingerprint,[]).append((index,dialogue))
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    n=max(1,int(len(keys)*val_fraction));val_keys=set(keys[:n])
    result={'train':[],'validation':[]};manifest=[]
    for key,entries in groups.items():
        part='validation' if key in val_keys else 'train'
        result[part].extend(entries)
        manifest.extend({'dialogue_index':index,'group_sha256':key,'split':part} for index,_ in entries)
    return result,manifest

def build_examples(entries):
    examples=[];excluded=[]
    for dialogue_index,dialogue in entries:
        history=[]
        for turn,msg in enumerate(dialogue.get('messages',[])):
            role=msg.get('role');text=msg.get('content','')
            if role not in ('user','assistant'):continue
            if not isinstance(text,str) or not text.strip():
                excluded.append({'dialogue_index':dialogue_index,'turn':turn,'reason':'empty_content'});continue
            if role=='user':
                gold=annotation(msg.get('annotation'))
                if gold:
                    target=f"认知偏差标签：{'、'.join(gold['tags'])}\n安全风险：{gold['risk']}\n偏差强度：{gold['intensity']}"
                    examples.append(dict(key=f'{dialogue_index}:{turn}:classify',dialogue_index=dialogue_index,turn=turn,
                        task='classify',history=list(history),user=text,target=target,gold=gold))
                elif 'annotation' in msg:
                    excluded.append({'dialogue_index':dialogue_index,'turn':turn,'reason':'invalid_annotation'})
            elif history and history[-1]['role']=='user':
                # Current user is already in history: remove it before rendering exactly once.
                examples.append(dict(key=f'{dialogue_index}:{turn}:generation',dialogue_index=dialogue_index,turn=turn,
                    task='generation',history=list(history[:-1]),user=history[-1]['content'],target=text))
            history.append({'role':role,'content':text})
    return examples,excluded

def prompt(example,c):
    classify=example['task']=='classify';kind='CLS' if classify else 'GEN'
    system=c['SYS_PROMPT_CLASSIFY' if classify else 'SYS_PROMPT_GENERATION']
    history=''.join(('患者：' if m['role']=='user' else '治疗师：')+m['content']+'\n' for m in example['history'])
    return f"{c[kind+'_TASK_ID']}\n{c[kind+'_MARKER']}\n{system}\n历史对话：\n{history}患者：{example['user']}"+c['RESPONSE_TEMPLATE']

def encode(example,tokenizer,c,max_length=1024,max_response=256):
    e=dict(example,history=list(example['history']))
    response=tokenizer.encode(e['target'],add_special_tokens=False)
    if not response or len(response)+1>max_response:raise ValueError('response_too_long_or_empty')
    response=response+[tokenizer.eos_token_id]
    dropped=0
    while True:
        prefix=tokenizer.encode(prompt(e,c),add_special_tokens=False)
        if len(prefix)+len(response)<=max_length:break
        if not e['history']:raise ValueError('instruction_and_current_turn_too_long')
        n=2 if len(e['history'])>1 and e['history'][0]['role']=='user' and e['history'][1]['role']=='assistant' else 1
        e['history']=e['history'][n:];dropped+=n
    # Exact generation prefix; labels can never fall back to whole-prompt supervision.
    return dict(input_ids=prefix+response,attention_mask=[1]*(len(prefix)+len(response)),
        labels=[-100]*len(prefix)+response,prompt_tokens=len(prefix),target_tokens=len(response),
        task=e['task'],key=e['key'],dropped_history_messages=dropped)

def collate(features,pad_token_id):
    import torch
    size=max(len(f['input_ids']) for f in features)
    batch={name:[] for name in ('input_ids','attention_mask','labels')}
    for f in features:
        padding=size-len(f['input_ids'])
        batch['input_ids'].append(f['input_ids']+[pad_token_id]*padding)
        batch['attention_mask'].append(f['attention_mask']+[0]*padding)
        batch['labels'].append(f['labels']+[-100]*padding)
        if not any(t!=-100 for t in f['labels']):raise ValueError('No response supervision')
    return {k:torch.tensor(v,dtype=torch.long) for k,v in batch.items()}
