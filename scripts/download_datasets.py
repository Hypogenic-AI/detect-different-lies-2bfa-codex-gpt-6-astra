"""Pinned direct parquet download, integrity and schema validation; run from repo root."""
from pathlib import Path
import requests, json, hashlib, concurrent.futures
import pyarrow.parquet as pq
root=Path.cwd();assert (root/'.git').is_dir();print(root)
specs=[('cais/MASK','4602b84dd9e2ca05c6e1eafbc14e556e908ac1bb','mask'),('mandarjoshi/trivia_qa','0f7faf33a3908546c6fd5b73a660e0f8ff173c2f','triviaqa')]
def download(t):
 ds,rev,local,name=t;p=root/'datasets'/local/name; print(root,flush=True);p.parent.mkdir(parents=True,exist_ok=True)
 url=f'https://huggingface.co/datasets/{ds}/resolve/{rev}/{name}'
 r=requests.get(url,timeout=180);r.raise_for_status();p.write_bytes(r.content)
 table=pq.read_table(p);rows=table.to_pylist();key='task_id' if local=='mask' else 'question_id'
 info=dict(dataset=ds,revision=rev,file=str(p.relative_to(root)),url=url,rows=len(rows),bytes=p.stat().st_size,sha256=hashlib.sha256(r.content).hexdigest(),schema=str(table.schema),nulls={k:table[k].null_count for k in table.column_names},unique_ids=len({str(x[key]) for x in rows}))
 if local=='triviaqa':info.update(empty_questions=sum(not x['question'] for x in rows),empty_answers=sum(not x.get('answer',{}).get('value') for x in rows),placeholder_answers=sum(x.get('answer',{}).get('value')=='<unk>' for x in rows),unique_questions=len({x['question'].strip().lower() for x in rows}))
 sample=root/'datasets'/'samples'/f'{local}_{p.parent.name}_{p.stem}.json';print(root,flush=True);sample.parent.mkdir(exist_ok=True);sample.write_text(json.dumps(rows[:3],indent=2));assert sample.stat().st_size<100000
 return info
jobs=[]
for ds,rev,local in specs:
 j=requests.get(f'https://huggingface.co/api/datasets/{ds}/revision/{rev}',timeout=60).json()
 for f in j['siblings']:
  n=f['rfilename']
  if n.endswith('.parquet') and (local=='mask' or n.startswith('rc.nocontext/')):jobs.append((ds,rev,local,n))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(download,jobs))
(root/'notes/dataset_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ['schema','url','nulls']} for r in out],indent=2))
