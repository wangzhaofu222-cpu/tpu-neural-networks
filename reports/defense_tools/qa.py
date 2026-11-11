from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import re,json,subprocess,hashlib
from PIL import Image,ImageOps,ImageDraw
from pypdf import PdfReader
from prepare import ROOT,OUT,TOOLS,sha,save,source

QA=ROOT/'tmp/pdfs/defense_20261111';QA.mkdir(parents=True,exist_ok=True)

def check():
    hashes=json.loads((OUT/'04_源码与核查/original_hashes.json').read_text('utf-8'))
    assert all(sha(ROOT/p)==v for p,v in hashes.items()),'Original reports changed'
    rows=[]
    alignment=json.loads((OUT/'04_源码与核查/translation_alignment.json').read_text('utf-8'))
    units=json.loads((TOOLS/'units.json').read_text('utf-8'))
    assert len(alignment)==len(units)==913
    mismatches=[]
    def numbers(s):
        s=re.sub(r'(?<=\d)[ \u00a0\u2009]+(?=\d)','',s)
        return re.findall(r'\d+(?:\.\d+)?',s)
    for r in alignment:
        a=units[r['unit']]['original'];b=r['translation']
        if numbers(a)!=numbers(b):mismatches.append({'unit':r['unit'],'original':a,'translation':b,'a':numbers(a),'b':numbers(b)})
    save(QA/'numeric_review.json',mismatches)
    for n in range(10):
        src=source(n);dst=OUT/'04_源码与核查'/f'lab{n}_zh'
        aa='\n'.join(p.read_text('utf-8-sig') for p in [src/'report_body.tex',*sorted((src/'tables').glob('*.tex'))])
        bb='\n'.join(p.read_text('utf-8-sig') for p in [dst/'report_body.tex',*sorted((dst/'tables').glob('*.tex'))])
        for pattern in [r'\\label\{[^}]*\}',r'\\(?:eqref|ref)\{[^}]*\}',r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}',r'\\input\{[^}]*\}',r'\\hypertarget\{[^}]*\}',r'\\run\{[^}]*\}']:
            assert Counter(re.findall(pattern,aa))==Counter(re.findall(pattern,bb)),(n,pattern)
        for p in (src/'figures').glob('*'):
            if p.is_file():assert sha(p)==sha(dst/'figures'/p.name),(n,p.name)
        log=(dst/'build/main.log').read_text('utf-8',errors='replace')
        errors=re.findall(r'Overfull[^\n]*|Missing character[^\n]*|[^\n]*undefined[^\n]*',log)
        assert not errors,(n,errors)
        ru=OUT/'01_俄语正式报告'/f'Lab{n}_Report.pdf';zh=OUT/'02_中文对照译本'/f'Lab{n}_中文对照译本.pdf'
        assert sha(ru)==sha(src/f'Lab{n}_Report.pdf')
        reader=PdfReader(zh)
        texts=[p.extract_text() for p in reader.pages]
        assert all(t.strip() for t in texts),(n,'blank page')
        assert not any('@@' in t or '\ufffd' in t for t in texts),(n,'bad text')
        rows.append({'lab':n,'ru_pages':len(PdfReader(ru).pages),'zh_pages':len(reader.pages),'ru_sha256':sha(ru),'zh_sha256':sha(zh),'figures':len(re.findall(r'\\includegraphics',aa)),'tables':len(re.findall(r'\\begin\{(?:table|longtable)\}',aa)),'russian_copy_identical':True,'references_and_figures_identical':True,'tex_warnings':errors})
    guide=OUT/'03_中俄答辩提纲/Lab0-9_中俄双语答辩指南.pdf'
    log=(OUT/'04_源码与核查/guide/build/main.log').read_text('utf-8',errors='replace')
    assert not re.search(r'Overfull|Missing character|undefined',log)
    assert len(re.findall(r'^### ',(TOOLS/'答辩提纲.md').read_text('utf-8'),re.M))==44
    data={'original_files_checked':len(hashes),'originals_unchanged':True,'translation_units':913,'translation_reports':rows,'guide_pages':len(PdfReader(guide).pages),'questions':44,'numeric_review_items':len(mismatches),'visual_review':'pending'}
    save(OUT/'04_源码与核查/qa_manifest.json',data)
    print(json.dumps(data,ensure_ascii=False,indent=2))

def render(p):
    label='guide' if p.parent.name.startswith('03_') else p.name.split('_')[0]
    dest=QA/(label+'_'+sha(p)[:10]);dest.mkdir(exist_ok=True)
    if len(list(dest.glob('page-*.png')))!=len(PdfReader(p).pages):
        subprocess.run(['pdftoppm','-r','100','-png',str(p),str(dest/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    pages=sorted(dest.glob('page-*.png'),key=lambda x:int(x.stem.split('-')[-1]));sheets=[]
    for start in range(0,len(pages),4):
        sheet=Image.new('RGB',(1440,2100),'#d8dce1');draw=ImageDraw.Draw(sheet)
        for i,page in enumerate(pages[start:start+4]):
            img=Image.open(page).convert('RGB');img.thumbnail((700,990))
            x=(i%2)*720+10;y=(i//2)*1050+30
            sheet.paste(img,(x,y));draw.text((x,y-23),f'{label}: page {start+i+1}',fill='black')
        q=dest/f'contact_{start//4+1:02}.jpg';sheet.save(q,quality=89);sheets.append(str(q))
    return label,sheets

if __name__=='__main__':
    check()
    pdfs=[*sorted((OUT/'02_中文对照译本').glob('*.pdf')),*list((OUT/'03_中俄答辩提纲').glob('*.pdf'))]
    rendered={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(render,p) for p in pdfs]):
            name,paths=f.result();rendered[name]=paths;print('Rendered',name,len(paths),'contact sheets',flush=True)
    save(QA/'render_index.json',rendered)
