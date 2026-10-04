"""Matched-question MASK persona-pressure feasibility check, separate from main QA.

Uses the factual neutral belief question in both arms, preserving original system
persona only for pressure. This is an adaptation, not the official MASK score.
"""
import json
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np
from common import ROOT, SYSTEM, read_jsonl, norm
from run_local import LocalModel

if __name__=='__main__':
    assert Path.cwd()==ROOT
    df=pd.read_parquet(ROOT/'datasets/mask/known_facts/test-00000-of-00001.parquet')
    df=df.iloc[np.random.default_rng(42).permutation(len(df))[:100]]
    items=[];tasks=[]
    for r in df.to_dict('records'):
        q=r['belief_elicit_1'].split('After thinking')[0].strip()
        q+=' Reply with only yes, no, or UNKNOWN.'
        items.append({**r,'matched_question':q})
        for kind,reps in [('belief',5),('neutral',1),('pressure',1)]:
            for repeat in range(reps):
                system=r['system_prompt'] if kind=='pressure' else SYSTEM
                tasks.append({'id':r['task_id']+f':{kind}:{repeat}','qid':r['task_id'],'kind':kind,'repeat':repeat,'messages':[{'role':'system','content':system},{'role':'user','content':q}]})
    (ROOT/'results/mask_items.json').write_text(json.dumps(items,indent=2))
    out=ROOT/'results/model_outputs/mask.jsonl';done={r['id'] for r in read_jsonl(out)} if out.exists() else set()
    tasks=[t for t in tasks if t['id'] not in done]
    lm=LocalModel();start=time.time()
    for offset in range(0,len(tasks),64):
        batch=lm.generate(tasks[offset:offset+64],80000+offset//64)
        with out.open('a') as f:
            for r in batch:f.write(json.dumps(r)+'\n')
        print('MASK',offset+len(batch),'/',len(tasks),flush=True)
    records=[r for r in read_jsonl(out) if r['kind'] in ['neutral','pressure']]
    xx=[];nn=[];ll=[]
    for offset in range(0,len(records),8):
        x,n,l=lm.features(records[offset:offset+8]);xx.append(x);nn.append(n);ll.append(l)
    np.savez_compressed(ROOT/'results/mask_features.npz',ids=np.array([r['id'] for r in records]),x=np.concatenate(xx),nll=np.concatenate(nn),lengths=np.concatenate(ll))
    (ROOT/'results/mask_timing.json').write_text(json.dumps({'seconds':time.time()-start,'n_questions':100},indent=2))
