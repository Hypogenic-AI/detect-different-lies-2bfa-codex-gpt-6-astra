from pathlib import Path
import json,hashlib,subprocess,datetime
import pyarrow.parquet as pq
from pypdf import PdfReader
root=Path.cwd();assert (root/'.git').is_dir();print(root)
checks=[]
def check(name,condition):
 checks.append(dict(check=name,passed=bool(condition)));assert condition,name
for f in ['papers/README.md','datasets/README.md','datasets/.gitignore','code/README.md','literature_review.md','resources.md','planning.md','notes/reading_notes.md','pyproject.toml','uv.lock']:
 check(f,Path(f).is_file() and Path(f).stat().st_size>0)
m=json.loads(Path('notes/paper_metadata.json').read_text())
check('ten PDF papers',len(m)==10)
for x in m:
 p=Path('papers')/(x['id']+'.pdf');check('PDF hash '+x['id'],hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']);check('PDF pages '+x['id'],len(PdfReader(p).pages)==x['pages'])
 chunks=list(Path('papers/pages').glob(x['id']+'_chunk_*.pdf'));check('all chunks '+x['id'],sum(len(PdfReader(c).pages) for c in chunks)==x['pages'])
v=json.loads(Path('notes/dataset_validation.json').read_text())
for x in v:
 p=Path(x['file']);check('data hash '+str(p),hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']);check('data rows '+str(p),pq.read_metadata(p).num_rows==x['rows']);check('git ignores '+str(p),subprocess.run(['git','check-ignore','-q',str(p)]).returncode==0)
check('MASK count1000',sum(x['rows'] for x in v if x['dataset']=='cais/MASK')==1000)
test=pq.read_table('datasets/triviaqa/rc.nocontext/test-00000-of-00001.parquet',columns=['answer']).to_pylist();check('test all placeholder',all(x['answer']['value']=='<unk>' for x in test))
for p in Path('datasets/samples').glob('*.json'):
 check('small valid sample '+str(p),len(json.loads(p.read_text()))==3 and p.stat().st_size<100000);check('sample git-visible '+str(p),subprocess.run(['git','check-ignore','-q',str(p)]).returncode==1)
for r in json.loads(Path('notes/code_manifest.json').read_text()):
 commit=subprocess.check_output(['git','-C',r['path'],'rev-parse','HEAD'],text=True).strip();check('commit '+r['name'],commit==r['commit']);check('clone ignored '+r['name'],subprocess.run(['git','check-ignore','-q',r['path']+'/README.md']).returncode==0)
check('probe submodule',subprocess.check_output(['git','-C','code/liars-bench/src/probes','rev-parse','HEAD'],text=True).strip()=='bb4ed7fb51e6d9b7abc4b089d04c9f5566e7bc8a')
check('Anthropic saved',Path('notes/anthropic_lie_detectors.html').stat().st_size>10000)
report=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,papers=10,datasets=2,root_repositories=5,probe_submodules=1,baseline_inference_run=False)
Path('notes/resource_validation.json').write_text(json.dumps(report,indent=2));print(f'Passed {len(checks)} resource checks.')
