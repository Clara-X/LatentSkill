import json, collections, re
from pathlib import Path
from transformers import AutoTokenizer
from evals.searchqa.skillrl_utils import em_check, normalize_searchqa_answer
root=Path(__file__).resolve().parents[1]
rows=[json.loads(s) for s in (root/'results/results_detail.jsonl').read_text().splitlines()]
meta=json.loads((root/'launch_evaluation.json').read_text())
inputs=[json.loads(s) for s in Path('data/search_test/search_test_all.jsonl').read_text().splitlines()]
assert {r['id'] for r in rows}=={r['id'] for r in inputs}
assert len(rows)==len(inputs)==len({r['id'] for r in rows})
assert all(r['em']==em_check(r['pred_answer'],r['golden_answers']) for r in rows)
tok=AutoTokenizer.from_pretrained('models/Qwen3-8B',local_files_only=True)
steps=[(r,s) for r in rows for s in r['steps']]
lengths=tok([s['output'] for r,s in steps],add_special_tokens=False,return_length=True,verbose=False)['length']
near=[(r,s,n) for (r,s),n in zip(steps,lengths) if n>=2040]
empty=[r for r in rows if not r['pred_answer'].strip()]
byds={}
for ds in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
 rr=[r for r in rows if r['dataset']==ds];n=len(rr);correct=sum(r['em'] for r in rr);paper=meta['paper_expected_em'][ds]
 byds[ds]=dict(count=n,correct=correct,em=correct/n,paper_em=paper,delta_pp=round((correct/n-paper)*100,2),empty_answers=sum(not r['pred_answer'].strip() for r in rr),average_steps=sum(len(r['steps']) for r in rr)/n,max_step_questions=sum(len(r['steps'])==4 for r in rr))
log=(root/'evaluation.log').read_text()
out=dict(count=len(rows),correct=sum(r['em'] for r in rows),em=sum(r['em'] for r in rows)/len(rows),datasets=byds,empty_answers=len(empty),empty_final_actions=dict(collections.Counter(r['steps'][-1].get('action') for r in empty)),actions=dict(collections.Counter(s.get('action') for r,s in steps)),retrieval_statuses=dict(collections.Counter(re.findall(r'\[RETRIEVE\] status=(\d+)',log))),near_generation_limit_steps=len(near),near_generation_limit_questions=len({r['id'] for r,s,n in near}),near_generation_limit_empty_questions=len({r['id'] for r,s,n in near if not r['pred_answer'].strip()}),empty_retrieval_steps=sum(s.get('action')=='search' and not s.get('retrieved_docs') for r,s in steps),average_steps=len(steps)/len(rows),checks='3125 unique expected IDs; EM recomputed and matched')
(root/'analysis/metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
