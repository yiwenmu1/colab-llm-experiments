import unittest
from data import build_examples,split_dialogues,encode
from cprl_contracts import current_utterance,transition,checked_embedding
class Tokenizer:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c)+1 for c in text]
class Tests(unittest.TestCase):
    def setUp(self):
        self.dialogue={'messages':[{'role':'user','content':'唯一当前发言','annotation':{'bias_tags':'灾难化','risk_level':'中危','bias_intensity':'中等'}},{'role':'assistant','content':'唯一目标回复'}]}
    def test_generation_no_target_leak_or_duplicate_user(self):
        samples,_=build_examples([(1,self.dialogue)])
        generation=next(x for x in samples if x['task']=='generation')
        self.assertEqual(generation['history'],[]);self.assertEqual(generation['user'],'唯一当前发言')
        self.assertNotIn('唯一目标回复',str(generation['history']))
    def test_duplicate_dialogues_same_partition(self):
        parts,manifest=split_dialogues([self.dialogue,self.dialogue,{'messages':[{'role':'user','content':'不同'}]}])
        self.assertEqual(manifest[0]['split'],manifest[1]['split'])
    def test_missing_annotation_not_negative(self):
        d={'messages':[{'role':'user','content':'未标注'},{'role':'assistant','content':'回复'}]}
        samples,_=build_examples([(0,d)])
        self.assertEqual([s['task'] for s in samples],['generation'])
    def test_mask_and_eos(self):
        e=build_examples([(0,self.dialogue)])[0][0]
        c={'CLS_TASK_ID':'C','GEN_TASK_ID':'G','CLS_MARKER':'c','GEN_MARKER':'g','SYS_PROMPT_CLASSIFY':'S','SYS_PROMPT_GENERATION':'S','RESPONSE_TEMPLATE':'R'}
        f=encode(e,Tokenizer(),c,1024,256)
        self.assertTrue(all(x==-100 for x in f['labels'][:f['prompt_tokens']]))
        self.assertEqual(f['labels'][f['prompt_tokens']:],f['input_ids'][f['prompt_tokens']:])
        self.assertEqual(f['labels'][-1],0)
        self.assertEqual(f['dialogue_index'],0)
        self.assertEqual(f['turn'],0)
        with self.assertRaises(ValueError):encode(e,Tokenizer(),c,1,256)
    def test_current_turn_never_previous_history(self):
        self.assertEqual(current_utterance({'context':'患者：旧发言','s_text':'患者当前发言：新发言\n认知偏差：灾难化'}),'新发言')
        with self.assertRaises(ValueError):current_utterance({'context':'患者：旧发言'})
    def test_next_state_uses_new_patient(self):
        gold={'bias_tags':'灾难化','risk_level':'中危','bias_intensity':'中等'}
        state={'current_utterance':'起始','history':[]}
        embed=lambda s:[len(s)]*1024
        a=transition(state,0,'医生','新患者甲',1,gold,embed)
        b=transition(state,0,'医生','更长的新患者乙',1,gold,embed)
        self.assertNotEqual(a['next_state']['embedding'],b['next_state']['embedding'])
        self.assertEqual(a['next_state']['history'][-1]['content'],'医生')
        self.assertIsNone(transition(state,0,'医生','',0,None,embed,True)['next_state'])
        with self.assertRaises(ValueError):transition(state,0,'医生','下一轮',1,None,embed)
        with self.assertRaises(ValueError):checked_embedding([float('nan')]*1024)
if __name__=='__main__':unittest.main()
