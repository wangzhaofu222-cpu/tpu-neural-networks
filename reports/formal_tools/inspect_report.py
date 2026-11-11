from common import *
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
import sys
n=int(sys.argv[1]);out=REPORTS/f'lab{n}'/'formal';pdf=out/f'Lab{n}_Report.pdf'
r=PdfReader(pdf);text=[p.extract_text() or '' for p in r.pages]
body=(out/'report_body.tex').read_text('utf-8')
body=re.sub(r'\\input\{(tables/[^}]+)\}',lambda m:(out/m[1]).read_text('utf-8'),body)
compact=lambda x:re.sub(r'\s+','',x)
cover=compact(text[0])
for s in [f'Лабораторная работа № {n}','Ван Чжаофу','Григорьев Д. С.','Нейронные сети','Томск','2026']:
    assert compact(s) in cover,s
assert all(s not in ''.join(text) for s in ['Друки','Детекция объектов на изображениях','Методы распознавания образов'])
log=(out/'build/main.log').read_text('utf-8',errors='replace')
warnings=re.findall(r'^.*(?:Overfull|Underfull|Missing character|LaTeX(?: Font)? Warning:|Package \S+ Warning:|^!).*$',log,re.M)
render=out/'build/qa'/('render-'+sha(pdf)[:12]);render.mkdir(exist_ok=True)
subprocess.run(['pdftoppm','-scale-to','1450','-png',str(pdf),str(render/'page')],check=True,stdout=subprocess.DEVNULL)
images=sorted(render.glob('page-*.png'))
assert len(images)==len(r.pages)
contacts=[]
for i in range(0,len(images),4):
    canvas=Image.new('RGB',(1680,2440),'#dddddd');draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(images[i:i+4]):
        im=Image.open(p).convert('RGB');im.thumbnail((820,1175))
        x=10+(j%2)*840;y=32+(j//2)*1210;canvas.paste(im,(x,y))
        draw.text((x,y-20),f'Lab {n} / page {i+j+1}',fill='black')
    dest=out/'build/qa'/f'contact-{i//4+1:02}.png';canvas.save(dest);contacts.append(str(dest))
data={'lab':n,'pages':len(r.pages),'figures':len(re.findall(r'\\begin\{figure\}',body)),
      'tables':len(re.findall(r'\\begin\{table\}',body))+len(re.findall(r'\\begin\{longtable\}',body)),
      'warnings':warnings,'contacts':contacts,'cover_fields_correct':True,'sha256':sha(pdf)}
save(out/'build/pdf_inspection.json',data)
update(n,pages=data['pages'],figures=data['figures'],tables=data['tables'])
print(json.dumps(data,ensure_ascii=False,indent=2))
