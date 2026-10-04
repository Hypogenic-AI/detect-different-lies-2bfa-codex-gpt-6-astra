"""Recreate pinned root repositories. Run from workspace root; no dependency installs."""
import json,subprocess
from pathlib import Path
assert (Path.cwd()/'.git').is_dir();print(Path.cwd())
for r in json.loads(Path('notes/code_manifest.json').read_text()):
 print(Path.cwd(),flush=True)
 if not Path(r['path']).exists():
  subprocess.run(['git','clone','--depth','1',r['url'],r['path']],check=True)
 subprocess.run(['git','-C',r['path'],'fetch','--depth','1','origin',r['commit']],check=True)
 subprocess.run(['git','-C',r['path'],'checkout','--detach',r['commit']],check=True)
print(Path.cwd(),flush=True)
subprocess.run(['git','-C','code/liars-bench','submodule','update','--init','--depth','1','src/probes'],check=True)
