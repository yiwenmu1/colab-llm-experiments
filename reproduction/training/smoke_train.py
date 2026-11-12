"""Real Qwen3-8B 20-step controls. No published adapter, no test set, no external LLM API."""
import argparse,hashlib,json,random,time,os
os.environ["USE_TF"]="0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
from data import split_dialogues,build_examples,templates,encode,collate
p=argparse.ArgumentParser();p.add_argument('--train-data',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=20)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
BASE='Qwen/Qwen3-8B';REVISION='b968826d9c46dd6066d109eabc6255188de91218'
raw=json.loads(a.train_data.read_text());splits,split_manifest=split_dialogues(raw)
c=templates(a.source);tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
(a.out/'split_manifest.json').write_text(json.dumps(split_manifest,indent=2))
prepared={};exclusions={}
for part,entries in splits.items():
    examples,excluded=build_examples(entries);encoded=[];counts={'classify':0,'generation':0}
    # Freeze a small pilot before any model results are inspected.
    random.Random(42).shuffle(examples)
    for e in examples:
        if counts[e['task']]>=32:continue
        try:
            encoded.append(encode(e,tokenizer,c,max_length=1024,max_response=256));counts[e['task']]+=1
        except ValueError as exc:excluded.append({'key':e['key'],'reason':str(exc)})
        if all(v>=32 for v in counts.values()):break
    prepared[part]=encoded;exclusions[part]=excluded
(a.out/'data_audit.json').write_text(json.dumps({'source_sha256':hashlib.sha256(a.train_data.read_bytes()).hexdigest(),
    'dialogues':{k:len(v) for k,v in splits.items()},'pilot_encoded_examples':{k:{t:sum(x['task']==t for x in v) for t in ('classify','generation')} for k,v in prepared.items()},
    'exclusions':exclusions},ensure_ascii=False,indent=2))
train={t:[e for e in prepared['train'] if e['task']==t][:16] for t in ('classify','generation')}
validation={t:[e for e in prepared['validation'] if e['task']==t][:16] for t in ('classify','generation')}
assert all(len(v)==16 for v in list(train.values())+list(validation.values()))
assert not ({e['dialogue_index'] for e in prepared['train']} & {e['dialogue_index'] for e in prepared['validation']})
(a.out/'pilot_manifest.json').write_text(json.dumps({'train':{t:[e['key'] for e in v] for t,v in train.items()},
    'validation':{t:[e['key'] for e in v] for t,v in validation.items()}},indent=2))
# Verify real tokenizer masking and padding before loading model weights.
b=collate([train['classify'][0],train['generation'][0]],tokenizer.pad_token_id)
assert torch.all(b['labels'][b['attention_mask']==0]==-100)
for i,e in enumerate([train['classify'][0],train['generation'][0]]):
    assert torch.all(b['labels'][i,:e['prompt_tokens']]==-100)
    assert b['labels'][i,len(e['input_ids'])-1].item()==tokenizer.eos_token_id
(a.out/'mask_check.json').write_text(json.dumps({'passed':True,'response_only':True,'padding_ignored':True,'true_eos_supervised':True},indent=2))
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(BASE,revision=REVISION,quantization_config=quant,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
model.config.use_cache=False
initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
(a.out/'config.json').write_text(json.dumps({'base':BASE,'revision':REVISION,'author_adapter_loaded':False,'arms':['generation_only','joint'],
    'steps_per_arm':a.steps,'micro_batch':1,'gradient_accumulation':1,'max_length':1024,'max_response':256,'lr':2e-4,
    'lora_r':16,'lora_alpha':32,'lora_dropout':.05,'quant_compute':'bf16','seed':42,'joint_sampling':'strict alternating 1:1',
    'scope':'small real-data pipeline test; not formal efficacy or PRO+DSCO reproduction','torch':torch.__version__,
    'gpu':torch.cuda.get_device_name(0),'gpu_memory':torch.cuda.get_device_properties(0).total_memory},indent=2))

def batch(e):return {k:v.to('cuda') for k,v in collate([e],tokenizer.pad_token_id).items()}
@torch.inference_mode()
def evaluate():
    model.eval();result={}
    for task,examples in validation.items():
        losses=[]
        for e in examples:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch(e)).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
            losses.append(float(loss))
        result[task]=sum(losses)/len(losses)
    return result
@torch.inference_mode()
def predict_validation(label):
    model.eval();rows=[]
    for e in validation['classify'][:8]:
        ids=torch.tensor([e['input_ids'][:e['prompt_tokens']]],device='cuda')
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=128,do_sample=False,use_cache=True,
                              pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        rows.append({'key':e['key'],'text':tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True),
                     'target':tokenizer.decode(e['input_ids'][e['prompt_tokens']:-1],skip_special_tokens=True),
                     'hit_limit':output.shape[1]-ids.shape[1]>=128})
    (a.out/(label+'_validation_predictions.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return rows

baseline=evaluate();predict_validation('initial')
result={'baseline_validation_loss':baseline,'arms':{}}
for arm in ('generation_only','joint'):
    set_peft_model_state_dict(model,initial)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=2e-4)
    scheduler=get_linear_schedule_with_warmup(optimizer,2,a.steps)
    histories=[];resume_verified=False
    for step in range(a.steps):
        task='generation' if arm=='generation_only' or step%2 else 'classify'
        e=train[task][(step//2 if arm=='joint' else step)%len(train[task])]
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch(e)).loss
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.0)
        if not torch.isfinite(norm) or float(norm)==0:raise RuntimeError('Gradient absent or non-finite')
        optimizer.step();scheduler.step()
        row={'arm':arm,'step':step+1,'task':task,'loss':float(loss),'gradient_norm':float(norm),'supervised_tokens':e['target_tokens']}
        histories.append(row)
        with (a.out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print('TRAIN',row,flush=True)
        if step+1==10:
            # Checkpoint serialization + same-process restoration; not a cold-start test.
            checkpoint=a.out/(arm+'_resume.pt')
            state={'adapter':{k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},
                   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step+1,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'random_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            torch.save(state,checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            with torch.no_grad():
                next(v for v in model.parameters() if v.requires_grad).add_(0.5)
            optimizer.param_groups[0]['lr']=0.123
            set_peft_model_state_dict(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['random_rng']);np.random.set_state(saved['numpy_rng'])
            resume_verified=all(torch.equal(v.detach().cpu(),saved['adapter'][k]) for k,v in get_peft_model_state_dict(model).items())
            assert resume_verified
            assert optimizer.param_groups[0]['lr']==saved['optimizer']['param_groups'][0]['lr']
            assert scheduler.state_dict()['last_epoch']==saved['scheduler']['last_epoch']
            checkpoint.unlink();del state,saved
    validation_loss=evaluate();predict_validation(arm)
    adapter_dir=a.out/arm/'adapter';model.save_pretrained(adapter_dir,safe_serialization=True)
    result['arms'][arm]={'optimizer_steps':a.steps,'tasks_seen':{t:sum(h['task']==t for h in histories) for t in ('classify','generation')},
        'validation_loss':validation_loss,'same_process_resume_verified':resume_verified,'finite_nonzero_gradients':True,
        'adapter_changed':any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in get_peft_model_state_dict(model).items())}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    del optimizer,scheduler
    torch.cuda.empty_cache()
print('SMOKE COMPLETE',json.dumps(result),flush=True)
