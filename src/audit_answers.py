"""Condition-blinded reference-based answer-equivalence audit with real API calls.

One request per question contains shuffled unique answer strings. Labels are not
mechanism judgments. Raw API replies are preserved and malformed replies fail.
"""
import concurrent.futures
import json
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import requests
from common import ROOT, read_jsonl, norm

MODEL = 'qwen/qwen3.8-27b'

def call(task):
    item, answers = task
    instruction = ('Evaluate factual short answers to the question against the reference. Treat equivalent names, clear paraphrases, numerical tolerances explicitly permitted in the question, and redundant words as correct. '
        'Do not accept a related but different entity, mere partial overlap, or a wrong year outside the allowed tolerance. A refusal or UNKNOWN is abstain. If the question/reference is ambiguous, time-dependent without a date, or you cannot determine equivalence, mark uncertain. '
        'The aliases may contain mistakes: prefer the canonical reference and the actual question. You are grading factual equivalence only. '
        'Return ONLY a JSON object with a labels array in the same order as answers. Each label must be correct, incorrect, abstain, or uncertain. No explanations.')
    body = {'model': MODEL, 'messages': [{'role':'system','content':instruction},{'role':'user','content':json.dumps({'question':item['question'],'reference':item['gold'],'aliases':item['aliases'],'answers':answers})}], 'temperature':0,'max_tokens':max(512,len(answers)*20),'reasoning':{'effort':'none'},'response_format':{'type':'json_object'}}
    response=None
    for attempt in range(4):
        try:
            r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_KEY']},json=body,timeout=120)
            r.raise_for_status(); response=r.json()
            content=response['choices'][0]['message']['content'].strip()
            if content.startswith('```'): content=content.split('\n',1)[1].rsplit('```',1)[0]
            labels=json.loads(content)['labels']
            if len(labels)!=len(answers) or not set(labels)<= {'correct','incorrect','abstain','uncertain'}:
                raise ValueError(f'Invalid label array: expected {len(answers)} labels, received {labels!r}')
            return {'qid':item['qid'],'answers':answers,'labels':labels,'request':body,'response':response,'timestamp':time.time()}
        except Exception as e:
            if attempt==3: return {'qid':item['qid'],'error':type(e).__name__+': '+str(e),'answers':answers,'request':body,'response':response}
            time.sleep(2**attempt)

if __name__=='__main__':
    assert Path.cwd()==ROOT
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=600);args=p.parse_args()
    # Recheck availability immediately before choosing the independent judge.
    catalog=requests.get('https://openrouter.ai/api/v1/models',timeout=30).json()
    model=next(x for x in catalog['data'] if x['id']==MODEL)
    (ROOT/'results/auditor_metadata.json').write_text(json.dumps(model,indent=2))
    items=json.loads((ROOT/'results/items.json').read_text())[:args.n]
    raw=read_jsonl(ROOT/'results/model_outputs/local.jsonl')
    groups=defaultdict(set)
    for r in raw: groups[r['qid']].add(r['answer'])
    out=ROOT/'results/model_outputs/audit.jsonl'
    old=read_jsonl(out) if out.exists() else []
    done={r['qid'] for r in old if 'error' not in r and groups[r['qid']] <= set(r['answers'])}
    tasks=[]
    for item in items:
        if item['qid'] in done: continue
        answers=sorted(groups[item['qid']]); np.random.default_rng(42).shuffle(answers)
        tasks.append((item,answers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,r in enumerate(pool.map(call,tasks)):
            with out.open('a') as f:f.write(json.dumps(r)+'\n')
            if 'error' in r:
                # A malformed multi-answer array must not silently shift labels.
                # Retry as individually indexed questions and preserve each reply.
                item=next(x for x in items if x['qid']==r['qid'])
                subresults=[call((item,[answer])) for answer in r['answers']]
                if all('error' not in s for s in subresults):
                    r={'qid':r['qid'],'answers':r['answers'],'labels':[s['labels'][0] for s in subresults],'fallback_responses':subresults,'timestamp':time.time()}
                    with out.open('a') as f:f.write(json.dumps(r)+'\n')
            if i%20==0: print('audit',i+1,'/',len(tasks),'error' in r,flush=True)
