"""Prepare a fixed sample with question and answer-entity split protection."""
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, FRAMES, BELIEFS, SYSTEM, norm, update_state

assert Path.cwd() == ROOT
rng = np.random.default_rng(42)
df = pd.read_parquet(ROOT / 'datasets/triviaqa/rc.nocontext/validation-00000-of-00001.parquet')
df['normalized_q'] = df.question.map(norm)
df = df.drop_duplicates('question_id').drop_duplicates('normalized_q')
sample = df.iloc[rng.permutation(len(df))[:600]]
items = []
for i, r in enumerate(sample.to_dict('records')):
    a = r['answer']
    entity = norm(a['value'])
    split_number = int(hashlib.sha256(entity.encode()).hexdigest()[:8], 16) % 10
    split = 'train' if split_number < 6 else ('validation' if split_number < 8 else 'test')
    items.append({'index': i, 'qid': r['question_id'], 'question': r['question'], 'gold': a['value'], 'aliases': sorted(set([a['value']] + list(a['aliases']) + list(a['normalized_aliases']))), 'entity': entity, 'split': split})
assert len({x['qid'] for x in items}) == 600
assert len({norm(x['question']) for x in items}) == 600
for a in ['train', 'validation', 'test']:
    for b in ['train', 'validation', 'test']:
        if a != b:
            assert not ({x['entity'] for x in items if x['split'] == a} & {x['entity'] for x in items if x['split'] == b})
(ROOT / 'results/items.json').write_text(json.dumps(items, indent=2))
(ROOT / 'results/prompts.json').write_text(json.dumps({'system': SYSTEM, 'frames': FRAMES, 'beliefs': BELIEFS}, indent=2))
meta = {'seed': 42, 'n_questions': 600, 'unique_validation_questions': len(df), 'splits': dict(Counter(x['split'] for x in items)), 'missing': int(sample.question.isna().sum()), 'python': platform.python_version(), 'knowledge_samples': 5, 'independent_entropy_samples': 3, 'temperature': .7, 'top_p': .9, 'max_new_tokens': 32, 'generation_batch': 64, 'activation_batch': 8, 'layers': [14, 28], 'dtype': 'bfloat16'}
(ROOT / 'results/config.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
update_state('Phase2 complete: fixed600-question sample, entity-disjoint splits, prompts/config saved in results/. Isolated dependencies installed; CUDA tensor smoke test passed. Pinned model revision in results/model_metadata.json; download ongoing. Next Phase3: implement resumable generation and activation extraction, then100-question pilot. No outcome data yet.')
