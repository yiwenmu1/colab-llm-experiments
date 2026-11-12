"""CPRL state contracts. External judgments are required; never synthesize success."""
import math,re
from data import annotation

def current_utterance(state):
    if isinstance(state.get('current_utterance'),str) and state['current_utterance'].strip():return state['current_utterance'].strip()
    match=re.search(r'患者当前发言[：:]\s*([^\n]+)',state.get('s_text',''))
    if match:return match[1].strip()
    raise ValueError('Current utterance absent; refusing previous-turn/default fallback')

def state_text(user,gold):
    a=annotation(gold)
    if not a:raise ValueError('Invalid next-state annotation')
    if not isinstance(user,str) or not user.strip():raise ValueError('Empty current utterance')
    return f"患者当前发言：{user}\n认知偏差：{'、'.join(a['tags'])}; 危险等级：{a['risk']}; 偏差强度：{a['intensity']}"

def checked_embedding(vector,dimension=1024):
    values=[float(x) for x in vector]
    if len(values)!=dimension or not all(math.isfinite(x) for x in values):raise ValueError('Invalid embedding')
    return values

def transition(state,action,doctor_reply,patient_reply,reward,next_annotation,embedder,terminal=False):
    if not 0<=int(action)<10 or int(action)!=action:raise ValueError('Invalid action')
    if not math.isfinite(float(reward)):raise ValueError('Invalid reward')
    history=list(state.get('history',[]))+[{'role':'user','content':current_utterance(state)},
        {'role':'assistant','content':doctor_reply}]
    next_state=None
    if not terminal:
        text=state_text(patient_reply,next_annotation)
        next_state={'current_utterance':patient_reply,'history':history,'s_text':text,'embedding':checked_embedding(embedder(text))}
    return {'state':state,'action':int(action),'reward':float(reward),'next_state':next_state,'terminal':bool(terminal)}
