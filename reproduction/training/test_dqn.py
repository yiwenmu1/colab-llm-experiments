"""Original DDQN/KL update on explicit synthetic unit-test fixtures, not efficacy data."""
import ast,json,random,tempfile
from collections import deque,namedtuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
source=Path('/Users/yiwenmu/Documents/ChatGPT/cbt/copollm-source/task2/dqn.py')
selected=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'QNetwork','ReplayBuffer','optimize_model'}]
ns=dict(torch=torch,nn=nn,F=F,np=np,random=random,deque=deque,Experience=namedtuple('Experience','state action reward next_state'),
    config=None,cp_config=None,device=torch.device('cpu'),EMBEDDING_DIM=1024,HIDDEN_DIM_1=256,HIDDEN_DIM_2=128,ACTION_DIM=10,
    BATCH_SIZE=32,GAMMA=.8,KL_BETA=.1,KL_TEMP=1.0)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
policy=ns['QNetwork'](1024,256,128,10);target=ns['QNetwork'](1024,256,128,10)
target.load_state_dict(policy.state_dict());target.eval()
optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
replay=ns['ReplayBuffer'](128)
for i in range(32):
 state=np.zeros(1024,dtype=np.float32);state[i]=1
 replay.push(state,i%10,float(i%2),None)
initial={k:v.clone() for k,v in policy.state_dict().items()}
losses=[ns['optimize_model'](policy,target,optimizer,replay)[0] for _ in range(30)]
assert all(np.isfinite(losses)) and losses[-1]<losses[0]
assert any(not torch.equal(initial[k],v) for k,v in policy.state_dict().items())
assert all(p.grad is None for p in target.parameters())
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'resume.pt';torch.save({'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},path)
 saved=torch.load(path,weights_only=False);other=ns['QNetwork'](1024,256,128,10);other.load_state_dict(saved['model'])
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in policy.state_dict().items())
print(json.dumps({'test':'original_ddqn_kl_synthetic_fixture_only','updates':30,'first_loss':losses[0],'last_loss':losses[-1],
    'policy_updated':True,'target_grad_absent':True,'all_terminal_batch_supported':True,'checkpoint_roundtrip':True,
    'clinical_or_policy_efficacy_evidence':False},indent=2))
