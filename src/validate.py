"""Validate completed research artifacts, split integrity and numerical claims."""
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from common import ROOT, read_jsonl, norm, score, update_state

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    assert Path.cwd()==ROOT
    checks={}
    items=json.loads((ROOT/'results/items.json').read_text())
    raw=read_jsonl(ROOT/'results/model_outputs/local.jsonl')
    checks['unique600_questions']=len(items)==len({i['qid'] for i in items})==len({norm(i['question']) for i in items})==600
    checks['unique11400_records']=len(raw)==len({r['id'] for r in raw})==11400
    counts=Counter(r['kind'] for r in raw)
    checks['generation_counts']=counts==Counter({'belief':3000,'verify':600,'entropy':1800,'neutral':600,'sham':600,'reward':600,'reputation':600,'instructed':600,'followup':3000})
    checks['entity_disjoint']=all(not ({i['entity'] for i in items if i['split']==a}&{i['entity'] for i in items if i['split']==b}) for a,b in [('train','test'),('train','validation'),('validation','test')])
    labels=pd.read_csv(ROOT/'results/labeled_outputs.csv');scores=pd.read_csv(ROOT/'results/scored_outputs.csv')
    checks['one_row_per_main_output']=len(labels)==len(scores)==3000 and set(labels.id)==set(scores.id)
    checks['factual_taxonomy_partition']=int(labels.false.sum())==int(labels.category.isin(['verified_misreport','uncertain_neutral_error','stable_wrong_belief','unresolved_false']).sum())
    summary=json.loads((ROOT/'results/summary.json').read_text())
    checks['shares_sum_to_one']=abs(sum(x['share'] for x in summary['shares_equal_four_arm_mixture'])-1)<1e-10
    checks['reported_arm_counts']=all(x['false']==int(labels[labels.kind.eq(x['arm'])].false.sum()) and x['correct']==int(labels[labels.kind.eq(x['arm'])].correct.sum()) for x in summary['arms'])
    meta=json.loads((ROOT/'results/replay_validation.json').read_text())
    checks['causal_prompt_feature']=meta['prompt_causal_check_pass']
    checks['greedy_replay_all10']=meta['matches']==meta['n']==10
    checks['score_empty_abstains']=score('',items[0])['abstain'] and not score('',items[0])['false']
    checks['score_reference_correct']=all(score(i['gold'],i)['correct'] for i in items)
    metrics=pd.read_csv(ROOT/'results/detector_metrics.csv')
    valid=metrics.dropna(subset=['auc'])
    checks['auc_valid_range']=bool(valid.auc.between(0,1).all() and valid.auc_low.between(0,1).all() and valid.auc_high.between(0,1).all())
    checks['undefined_auc_not_fabricated']=bool(metrics[(metrics.positive==0)|(metrics.negative==0)].auc.isna().all())
    api=read_jsonl(ROOT/'results/model_outputs/api.jsonl')
    checks['api400_successful']=len(api)==400 and not any('error' in r for r in api)
    checks['mask700_records']=len(read_jsonl(ROOT/'results/model_outputs/mask.jsonl'))==700
    audited=pd.read_csv(ROOT/'results/audited_labels.csv')
    checks['audit_covers_all_main']=len(audited)==3000 and set(audited.id)==set(labels.id)
    required=['REPORT.md','README.md','planning.md','results/summary.json','results/audited_summary.json','results/mask_summary.json','results/determinism_check.json','figures/response_categories.png','figures/detector_specificity.png','figures/entropy_by_state.png']
    checks['required_artifacts']=all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
    determinism=json.loads((ROOT/'results/determinism_check.json').read_text())
    checks['deterministic_analysis']=determinism['all_equal']
    report=(ROOT/'REPORT.md').read_text()
    checks['report_sections']=all(x in report for x in ['Executive Summary','Methodology','Results','Limitations','References'])
    env={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','pandas','scipy','scikit-learn','matplotlib','accelerate','requests']}}
    (ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py'))}
    manifest.update({p:sha(ROOT/p) for p in ['uv.lock','pyproject.toml','results/items.json','results/prompts.json','results/model_metadata.json']})
    (ROOT/'results/reproduction_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'results/validation.json').write_text(json.dumps({'checks':checks,'passed':sum(checks.values()),'total':len(checks),'all_passed':all(checks.values())},indent=2))
    print(json.dumps(checks,indent=2))
    assert all(checks.values()), [k for k,v in checks.items() if not v]
