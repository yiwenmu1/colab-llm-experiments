"""Generate a small, reviewable patch; preserve the upstream checkout unchanged."""
import ast,difflib
from pathlib import Path
root=Path('/Users/yiwenmu/Documents/ChatGPT/cbt/copollm-source')
path=root/'task4/sft_unified.py';old=path.read_text();new=old
needle='''# 构造输入：系统提示(生成) + 历史对话（截止到上一条user消息） + 当前user发言
                    input_msgs = [{"role": "system", "content": SYS_PROMPT_GENERATION}]
                    input_msgs.extend(history)'''
assert new.count(needle)==1
new=new.replace(needle,needle.replace('input_msgs.extend(history)','input_msgs.extend(history[:-1])'))
new=new.replace('default=120, help="最大训练步数"','default=-1, help="最大训练步数；-1 使用 epochs"')
start=new.index('            if not self.use_masking or not features:');end=new.index('            # 基本的 instruction masking 实现',start)
new=new[:start]+'''            if not self.use_masking or not features:
                raise ValueError("Completion-only masking unavailable; refusing full-prompt training")

'''+new[end:]
start=new.index('            for i, input_ids in enumerate(batch["input_ids"]):');end=new.index('            batch["labels"] = labels',start)
new=new[:start]+'''            for i, input_ids in enumerate(batch["input_ids"]):
                matches = [j for j in range(len(input_ids)-len(self.response_token_ids)+1)
                           if input_ids[j:j+len(self.response_token_ids)].tolist()==self.response_token_ids]
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous response boundary")
                labels[i, :matches[0]+len(self.response_token_ids)] = -100
                labels[i, batch["attention_mask"][i]==0] = -100

'''+new[end:]
needle='    training_arguments = SFTConfig(**training_arguments_dict)'
new=new.replace(needle,'''    import inspect
    supported = inspect.signature(SFTConfig).parameters
    length_key = "max_length" if "max_length" in supported else "max_seq_length" if "max_seq_length" in supported else None
    if length_key is None:
        raise RuntimeError("Installed SFTConfig exposes no supported sequence-length field")
    training_arguments_dict[length_key] = args.max_length
'''+needle)
start=new.index('    except Exception as e:\n        print(f"❌ DataCollatorForCompletionOnlyLM 初始化失败:')
end=new.index('\n\n',new.index('print("   🐛 Debug: 将对整个序列计算损失（包括指令和响应）")',start))
new=new[:start]+'''    except Exception as e:
        raise RuntimeError("Completion-only collator initialization failed; training stopped") from e'''+new[end:]
ast.parse(new)
# Reproduce the generator bug using the patched original builder on actual public data.
import json,random,io,contextlib
from data import templates
class Dataset:
    @staticmethod
    def from_list(rows):return rows
ns={'json':json,'random':random,'Dataset':Dataset,**templates(path)}
fn=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='build_mixed_dataset')
fn.returns=None
for arg in fn.args.args:arg.annotation=None
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<patched-upstream>','exec'),ns)
with contextlib.redirect_stdout(io.StringIO()):rows=ns['build_mixed_dataset'](str(root/'data/train.json'),None)
generation=[r for r in rows if r['task_type']=='generation']
duplicates=sum(len(r['messages'])>=3 and r['messages'][-2]['role']==r['messages'][-3]['role']=='user' and r['messages'][-2]['content']==r['messages'][-3]['content'] for r in generation)
assert duplicates==0
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/task4/sft_unified.py',tofile='b/task4/sft_unified.py'))
Path(__file__).with_name('upstream_sft_fixes.patch').write_text(patch)
print('Patched original builder:',len(generation),'generation samples; duplicate current-user count:',duplicates)
