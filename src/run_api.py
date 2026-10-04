"""Separate real OpenRouter behavioral replication; never pooled with local data."""
import concurrent.futures
import json
import os
import time
from pathlib import Path
import requests
from common import ROOT, messages, score, read_jsonl

def call(task):
    item, kind = task
    body = {'model': 'qwen/qwen-2.5-7b-instruct', 'messages': messages(item['question'], kind), 'temperature': .7, 'top_p': .9, 'max_tokens': 32, 'seed': 42, 'provider': {'require_parameters': True}}
    for attempt in range(4):
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_KEY']}, json=body, timeout=90)
            r.raise_for_status()
            response = r.json()
            answer = response['choices'][0]['message']['content']
            return {'id': item['qid'] + ':' + kind, 'qid': item['qid'], 'kind': kind, 'request': body, 'response': response, 'answer': answer, 'timestamp': time.time(), **score(answer, item)}
        except Exception as e:
            if attempt == 3: return {'id': item['qid']+':'+kind, 'error': str(e), 'timestamp': time.time()}
            time.sleep(2**attempt)

if __name__ == '__main__':
    assert Path.cwd() == ROOT
    items = json.loads((ROOT / 'results/items.json').read_text())[:100]
    out = ROOT / 'results/model_outputs/api.jsonl'
    done = {r['id'] for r in read_jsonl(out)} if out.exists() else set()
    tasks = [(i,k) for i in items for k in ['neutral','sham','reward','reputation'] if i['qid']+':'+k not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for j,r in enumerate(pool.map(call,tasks)):
            with out.open('a') as f: f.write(json.dumps(r)+'\n')
            if j%20 == 0: print('API', j+1, '/', len(tasks), 'error' in r, flush=True)
