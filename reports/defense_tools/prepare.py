"""Create isolated defense materials; original reports/evidence are read-only."""
from pathlib import Path
import json,re,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'答辩资料_Lab0-9_20261111'
TOOLS=Path(__file__).resolve().parent
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
def source(n):return ROOT/f'reports/lab{n}'/('formal' if n else '')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    for d in ['01_俄语正式报告','02_中文对照译本','03_中俄答辩提纲','04_源码与核查']:(OUT/d).mkdir(exist_ok=True)
    units={};inputs={}
    for n in range(10):
        src=source(n);pdf=src/f'Lab{n}_Report.pdf';inputs[str(pdf.relative_to(ROOT))]=sha(pdf)
        shutil.copy2(pdf,OUT/'01_俄语正式报告'/pdf.name)
        target=OUT/'04_源码与核查'/f'lab{n}_zh';target.mkdir(exist_ok=True)
        for folder in ['figures','assets']:
            if (src/folder).exists():shutil.copytree(src/folder,target/folder,dirs_exist_ok=True)
        tabledir=target/'tables';tabledir.mkdir(exist_ok=True)
        files=[src/'report_body.tex',*sorted((src/'tables').glob('*.tex'))]
        serial=0
        for p in files:
            rel=p.relative_to(src).as_posix();inputs[str(p.relative_to(ROOT))]=sha(p)
            # Preserve RUN_ID appendix rows exactly; translate their headings.
            text=p.read_text('utf-8-sig');lines=text.splitlines();records=[]
            for i,line in enumerate(lines):
                if re.search('[А-Яа-яЁё]',line):
                    protected=[]
                    def hide(m):
                        protected.append(m[0]);return f'@@{len(protected)-1}@@'
                    # Protect math, references and literal identifiers from translation.
                    line2=re.sub(r'\$[^$]*\$|\\(?:ref|eqref|label|cite|run|url|hypertarget|texttt)\{[^}]*\}',hide,line)
                    key=f'{n}.{serial:03}';serial+=1
                    units[key]={'lab':n,'file':rel,'line':i,'original':line,'prompt':line2,'protected':protected}
                    records.append(key)
            # Template remains separate; only translated copies are authored later.
            (target/(rel+'.original')).parent.mkdir(exist_ok=True,parents=True)
            (target/(rel+'.original')).write_text(text,'utf-8')
        print(n,serial)
    save(TOOLS/'units.json',units);save(OUT/'04_源码与核查'/'original_hashes.json',inputs)
    for n in range(10):
        text='\n'.join(k+'\t'+v['prompt'] for k,v in units.items() if v['lab']==n)
        (TOOLS/f'units_{n}.txt').write_text(text,'utf-8')
if __name__=='__main__':prepare()
