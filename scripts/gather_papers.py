"""Restore exactly the reviewed PDF revisions using the checked manifest."""
import concurrent.futures,hashlib,json
from pathlib import Path
import requests
from pypdf import PdfReader
root=Path.cwd();assert (root/'.git').is_dir();print(root)
records=json.loads((root/'notes/paper_metadata.json').read_text())
def download(r):
 print(root,flush=True)
 response=requests.get(r['source'],timeout=180);response.raise_for_status();data=response.content
 assert data.startswith(b'%PDF')
 assert hashlib.sha256(data).hexdigest()==r['sha256'], 'PDF bytes changed: '+r['id']
 p=root/'papers'/(r['id']+'.pdf');p.write_bytes(data)
 assert len(PdfReader(p).pages)==r['pages']
 return r['id']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 print(list(pool.map(download,records)))
