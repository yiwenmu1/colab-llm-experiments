"""Verify full prediction coverage and write an evidence-bounded Markdown report."""
import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results_dir',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
name='qwen3_8b_nf4_ctx2048_v1'
rows=[json.loads(l) for l in (a.results_dir/(name+'_diagnosis.jsonl')).read_text().splitlines() if l.strip()]
metrics=json.loads((a.results_dir/(name+'_metrics.json')).read_text())
manifest=json.loads((a.results_dir/'sample_manifest.json').read_text())
config=json.loads((a.results_dir/(name+'_manifest.json')).read_text())
expected={x['key']:x for x in manifest}
assert len(expected)==981
assert len(rows)==1962, f'Incomplete predictions: {len(rows)}/1962'
assert len({(r['key'],r['condition']) for r in rows})==1962, 'Duplicate keys'
for condition in ('base','copollm'):
 rr=[r for r in rows if r['condition']==condition]
 assert {r['key'] for r in rr}==set(expected), 'Missing/foreign keys'
 assert metrics[condition]['n']==981 and metrics[condition]['high_risk_n']==42
 for r in rr:
  gold=expected[r['key']]
  for key in ('gold_tags','gold_risk','gold_intensity'):
   assert r[key]==gold[key], f'Gold mismatch: {r["key"]}'
 classes=Counter(t for r in rr for t in r['gold_tags'])
 for label,value in metrics[condition]['per_label'].items():
  assert value['support']==classes[label]
 lines=981
text='''# CoPoLLM Qwen3-8B 公开子集分类复现报告

本报告覆盖作者公开 LoRA 与关闭 adapter 的 Qwen3-8B 基座在同一公开测试子集上的分类对照。
未训练 CPRL、PRO 或 DSCO，未调用教师 API，未评测成批回复生成质量。不能视为论文原测试集效果或完整训练方法复现。

## 数据与完整性

测试对话 324；有效标注 981；排除无效标注 41；高危 42。
两组共 1962 条预测，逐一核对样本唯一键、金标和类别支持数，覆盖完整。
未标注发言不作为无偏差负样本；公开有效样本无法完整检验多标签或无偏差识别能力。

## 固定协议

Qwen/Qwen3-8B；NF4 4-bit、double quant、FP16 计算；batch=1；输入上限 2048；输出上限 128；greedy。
两组使用仓库同一任务模板与 tokenizer；基座为关闭作者 adapter 的受控对照，不等同于其官方 chat template 最优效果。
输入保留当前发言和此前历史，不纳入当前目标回复或金标 annotation。超长历史按轮次移除。
沿用首次 v1 协议，没有根据测试分数调整提示或输出长度。

## 指标

| 指标 | 基座 | 作者 LoRA |
|---|---:|---:|
'''
for key,title in [('macro_precision','Macro Precision'),('macro_recall','Macro Recall'),('macro_f1','Macro F1'),('micro_f1','Micro F1'),('HRMDR','HRMDR'),('parse_failure_rate','格式失败率'),('input_trim_rate','历史裁剪率'),('generation_limit_rate','达到生成上限比例')]:
 text+=f'| {title} | {metrics["base"][key]:.6f} | {metrics["copollm"][key]:.6f} |\n'
text+='\n| 类别 | 支持数 | 基座 F1 | 作者 LoRA F1 |\n|---|---:|---:|---:|\n'
for label,b in metrics['base']['per_label'].items():
 c=metrics['copollm']['per_label'][label]
 text+=f'| {label} | {b["support"]} | {b["f1"]:.6f} | {c["f1"]:.6f} |\n'
delta=metrics['copollm']['macro_f1']-metrics['base']['macro_f1']
text+=f'\n作者 LoRA 相对基座的 Macro F1 差值为 {delta:+.6f}。这是当前协议下的描述性比较，未进行显著性推断。\n'
text+='''
## 解释边界与问题

- 基座小样本输出存在分类字段后继续解释、重复任务提示的行为。达到 token 上限不能单独证明分类字段缺失；上表同时保留格式失败与长度上限指标。
- HRMDR 是金标高危样本中未预测为高危的比例（越低越好），仅 42 条，每漏一条约改变 2.38 个百分点。
- Macro F1 为固定八类的多标签指示矩阵宏平均。格式失败不能静默剔除；解析得到的标签保留计分。
- 公开子集、量化、模板和上下文条件与论文原评测可能不同，不用论文参考值作为本子集验收线。
- v1 前 10 条来自此前 A100 运行，其余来自本轮续跑；模型 revision 与推理协议保持一致，软件/硬件记录以各轮 provenance 为准。跨轮数值完全一致性未另行验证。

## 可追溯配置

'''
text+='```json\n'+json.dumps(config,ensure_ascii=False,indent=2)+'\n```\n'
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(text)
print('Verified 1962 unique predictions. Report:',a.out)
