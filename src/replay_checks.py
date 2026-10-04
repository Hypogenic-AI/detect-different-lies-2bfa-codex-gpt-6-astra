"""Real-model replay and a causal prompt-token control check."""
import json
from pathlib import Path
import numpy as np
from common import ROOT, read_jsonl
from run_local import LocalModel

if __name__=='__main__':
    assert Path.cwd()==ROOT
    raw=read_jsonl(ROOT/'results/model_outputs/local.jsonl')
    sample=[r for r in raw if r['kind']=='verify'][:10]
    lm=LocalModel()
    replay=lm.generate(sample,999999)
    comparisons=[{'id':a['id'],'old':a['answer'],'new':b['answer'],'identical':a['answer']==b['answer']} for a,b in zip(sample,replay)]
    a=next(r for r in raw if r['kind']=='neutral' and r['correct'] and r['token_ids'])
    b=next(r for r in raw if r['kind']=='instructed' and r['token_ids']!=a['token_ids'])
    # Same factual prompt, two real saved continuation token sequences.
    modified={**a,'token_ids':b['token_ids']}
    x,_,_=lm.features([a,modified])
    prompt_difference=float(np.abs(x[0,:,1].astype(float)-x[1,:,1].astype(float)).max())
    output={'greedy_replay':comparisons,'matches':sum(r['identical'] for r in comparisons),'n':len(comparisons),'same_prompt_different_continuation_max_abs_feature_difference':prompt_difference,'prompt_causal_check_pass':prompt_difference<.02}
    (ROOT/'results/replay_validation.json').write_text(json.dumps(output,indent=2))
    assert output['prompt_causal_check_pass']
    print(json.dumps(output,indent=2))
