from pathlib import Path
import subprocess,fitz,json
root=Path.cwd();assert (root/'.git').is_dir();print(root)
(root/'notes/chunks').mkdir(exist_ok=True);(root/'notes/renders').mkdir(exist_ok=True)
for p in sorted((root/'papers').glob('*.pdf')):
 subprocess.run([str(root/'.venv/bin/python'),'.codex/skills/paper-finder/scripts/pdf_chunker.py',str(p),'--pages-per-chunk','6'],check=True,stdout=subprocess.DEVNULL)
 for c in sorted((root/'papers/pages').glob(p.stem+'_chunk_*.pdf')):
  d=fitz.open(c)
  (root/'notes/chunks'/f'{c.stem}.txt').write_text('\n'.join(f'--- PAGE {j+1} OF CHUNK ---\n'+page.get_text() for j,page in enumerate(d)))
  out=fitz.open();sheet=out.new_page(width=1836,height=1584)
  for j in range(len(d)):
   rect=fitz.Rect((j%3)*612,(j//3)*792,(j%3+1)*612,(j//3+1)*792)
   sheet.show_pdf_page(rect,d,j)
  sheet.get_pixmap().save(root/'notes/renders'/f'{c.stem}.png')
print([(p.name,len(fitz.open(p))) for p in (root/'papers').glob('*.pdf')])
