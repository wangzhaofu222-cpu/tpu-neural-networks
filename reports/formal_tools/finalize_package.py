"""Read-only evidence audit and independent, report-only compilation/package QA."""
from common import *
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import zipfile, statistics, csv

audit={'created':datetime.now().isoformat(), 'scope':'Reports only; no model execution', 'reports':{}}
tracked=load(REPORTS/'formal_inputs_sha256.json')
changed=[p for p,h in tracked.items() if sha(ROOT/p)!=h]
assert not changed,changed
audit['original_input_hashes']={'checked':len(tracked),'changed':changed}
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head=='ec08a6164d76f66f0a342b26a1dc3996ae23ec17',head
audit['git_HEAD_unchanged']=head
diff=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(diff)<= {'README.md','reports/lab0/main.tex'},diff  # README adds the current-state pointer; Lab0 change predates this task.
audit['tracked_diff']=diff

def embedded_fonts(reader):
    seen=set(); fonts={}
    def walk(res):
        if not res:return
        res=res.get_object()
        for name,ref in res.get('/Font',{}).items():
            f=ref.get_object(); key=str(f.get('/BaseFont',name))
            children=f.get('/DescendantFonts',[f])
            for child in children:
                c=child.get_object(); desc=c.get('/FontDescriptor')
                if c.get('/Subtype')=='/Type3': ok=True
                else:
                    desc=desc.get_object() if desc else {}
                    ok=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
                fonts[key]=fonts.get(key,True) and ok
        for ref in res.get('/XObject',{}).values():
            obj=ref.get_object(); ident=id(obj)
            if ident in seen:continue
            seen.add(ident);walk(obj.get('/Resources'))
    for p in reader.pages:walk(p.get('/Resources'))
    assert fonts and all(fonts.values()),fonts
    return fonts

for n in range(1,10):
    out=REPORTS/f'lab{n}'/'formal'; pdf=out/f'Lab{n}_Report.pdf'
    inspection=load(out/'build/pdf_inspection.json')
    assert sha(pdf)==inspection['sha256']==sha(out.parent/pdf.name)
    assert state()[str(n)]['visual']=='PASS'
    r=PdfReader(pdf); texts=[p.extract_text() or '' for p in r.pages]
    assert len(r.pages)==inspection['pages']
    for i,p in enumerate(r.pages):
        w,h=float(p.mediabox.width),float(p.mediabox.height)
        assert abs(w-595.28)<2 and abs(h-841.89)<2,(n,i,w,h)
        if i:assert re.search(r'\b'+str(i+1)+r'\s*$',texts[i]),(n,i,'page number')
    body=(out/'report_body.tex').read_text('utf-8')
    questions=re.findall(r'\\subsection\*\{(?:Вопрос )?([1-5])\.',body)
    assert questions==['1','2','3','4','5'],(n,questions)
    for p in out.rglob('*.tex'):
        if 'build' in p.parts:continue
        text=p.read_text('utf-8-sig')
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in text),str(p)
        assert '\t' not in text,('unexpected tab',str(p))
    bad=[w for w in inspection['warnings'] if 'Underfull' not in w]
    assert not bad,(n,bad)
    audit['reports'][str(n)]={**inspection,'fonts_embedded':embedded_fonts(r),
        'all_pages_visually_reviewed':True,'counterfactuals':state()[str(n)]['counterfactuals']}

# Recalculate paired statistics directly from archived scalar outputs, without model inference.
def csvrows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
p6=csvrows('experiments/lab6/E3_pairs.csv')
d6=[float(x['EBM'])-float(x['Flow']) for x in p6]
assert all(abs(d-float(x['delta']))<1e-14 for d,x in zip(d6,p6))
m9=load(ROOT/'experiments/lab9/E1_main.json')
d9=[x['F']['macro_f1']-x['U']['macro_f1'] for x in m9['rows']]
audit['recalculated_statistics']={}
for lab,d in [(6,d6),(9,d9)]:
    mean=statistics.mean(d);sd=statistics.stdev(d)
    established=(all(v>0 for v in d) or all(v<0 for v in d)) and abs(mean)>=2*sd
    audit['recalculated_statistics'][str(lab)]={'paired_differences':d,'mean':mean,'sample_sd':sd,'difference_established':established}
assert audit['recalculated_statistics']['6']['difference_established']
assert not audit['recalculated_statistics']['9']['difference_established']

# PATH-first build scripts. This only changes the nine new report projects.
template=r'''param()
$ErrorActionPreference = 'Stop'
$oldPath = $env:Path
$oldLocale = $env:LC_ALL
$oldLang = $env:LANG
Push-Location -LiteralPath $PSScriptRoot
try {
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    if (-not $latexmk) {
        $localTex = 'D:\Tex Live\data\texlive\2026\bin\windows'
        if (Test-Path -LiteralPath "$localTex\latexmk.exe") {
            $env:Path = $localTex + ';' + $env:Path
            $latexmk = Get-Command latexmk -ErrorAction Stop
        } else { throw 'Install TeX Live with XeLaTeX and latexmk, and add its bin directory to PATH.' }
    }
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    & $latexmk.Source -xelatex -interaction=nonstopmode -halt-on-error -outdir=build title_page.tex
    if ($LASTEXITCODE -ne 0) { throw 'Title-page compilation failed.' }
    Copy-Item -LiteralPath 'build\title_page.pdf' -Destination 'title_page.pdf'
    & $latexmk.Source -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Report compilation failed.' }
    Copy-Item -LiteralPath 'build\main.pdf' -Destination 'LabNUMBER_Report.pdf'
} finally {
    $env:Path = $oldPath
    $env:LC_ALL = $oldLocale
    $env:LANG = $oldLang
    Pop-Location
}
'''
stage=REPORTS/'package_validation'/datetime.now().strftime('%Y%m%d_%H%M%S')
stage.mkdir(parents=True,exist_ok=False)
for n in range(1,10):
    out=REPORTS/f'lab{n}'/'formal';target=stage/f'lab{n}';target.mkdir()
    (out/'build.ps1').write_text(template.replace('NUMBER',str(n)),encoding='utf-8-sig')
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.tex','.ps1','.pdf']:
            shutil.copy2(p,target/p.name)
    for name in ['assets','figures','tables']:shutil.copytree(out/name,target/name)

def rebuild(n):
    target=stage/f'lab{n}'
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(target/'build.ps1')],cwd=target,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'rebuild-console.log').write_bytes(result.stdout)
    assert result.returncode==0,(n,result.stdout.decode('utf-8','replace')[-2000:])
    rebuilt=PdfReader(target/f'Lab{n}_Report.pdf')
    original=PdfReader(REPORTS/f'lab{n}'/f'Lab{n}_Report.pdf')
    assert [p.extract_text() for p in rebuilt.pages]==[p.extract_text() for p in original.pages],n
    log=(target/'build/main.log').read_text('utf-8',errors='replace')
    assert not any(x in log for x in ['Overfull','Missing character','Undefined control sequence','undefined references']),n
    return n

print('Input/PDF audit passed. Independent clean builds starting.',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    for f in as_completed([pool.submit(rebuild,n) for n in range(1,10)]):
        n=f.result();audit['reports'][str(n)]['independent_clean_build']='PASS'
        print(f'Lab {n}: independent clean build PASS',flush=True)

readme='''# Lab 1–9 — официальные отчёты / 正式报告

В каждой папке labN находятся PDF, main.tex, report_body.tex, title_page.tex,
title_page.pdf, assets/, figures/, tables/ и build.ps1. Все зависимости проекта
локальны. Исходные модели и датасеты для компиляции не требуются.

Сборка: откройте папку labN в PowerShell и выполните .\\build.ps1.
Нужны TeX Live (XeLaTeX, latexmk), шрифты Times New Roman, Arial, Consolas,
SimSun. При отсутствии этих шрифтов сначала установите их законным способом.
Скрипт ищет latexmk в PATH, затем в исходном локальном расположении TeX Live.
Для другой ОС: скомпилируйте title_page.tex, затем main.tex с XeLaTeX;
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build.
PDF обложки из build/title_page.pdf нужно скопировать в title_page.pdf перед
сборкой main.tex. Готовый PDF находится в build/main.pdf.

全部九份均已从本压缩包对应的独立目录重新编译验证，122页以下以清单为准。
页数含封面、目录、文献和运行编号附录；图数按编号图计（多面板算一图），
表数含 RUN_ID 附录表。报告基于真实结果；2026-11-06 版新增 Lab1 E2
原条件复跑及 Lab4 色调控制的六次线性探针训练，原实验记录保留。
Lab4 第2反事实问题已补齐色调控制实验。Lab5/9 使用已选定的项目协议。
Lab 8 E2 保留与假设相反的结果；Lab 9 保留“差异未确立”的统计结论。
'''.replace('122页以下以清单为准','总页数以清单为准')
readme+='\n| Lab | Страниц | Рисунков | Таблиц | Контрфактические вопросы |\n|---|---:|---:|---:|---|\n'
for n in range(1,10):
    s=state()[str(n)]
    readme+=f"| {n} | {s['pages']} | {s['figures']} | {s['tables']} | {s['counterfactuals']} |\n"
(stage/'README.md').write_text(readme,'utf-8')
(REPORTS/'README_Lab1-9.md').write_text(readme,'utf-8')
dest=REPORTS/'Lab1-9_Reports_20261106.zip'
assert not dest.exists(),'Do not overwrite an existing archive without inspection.'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    z.write(stage/'README.md','README.md')
    for n in range(1,10):
        out=REPORTS/f'lab{n}'/'formal'
        for p in out.rglob('*'):
            rel=p.relative_to(out)
            if not p.is_file() or 'build' in rel.parts:continue
            assert p.suffix in ['.tex','.ps1','.pdf','.png','.svg','.jpg','.jpeg'],str(p)
            z.write(p,(Path(f'lab{n}')/rel).as_posix())
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    names=z.namelist()
    assert len([p for p in names if re.search(r'/Lab\d_Report.pdf$',p)])==9
    assert len([p for p in names if p.endswith('/main.tex')])==9
    assert not any('/build/' in p or p.endswith(('.pt','.pth','.log','.aux','.pyc')) for p in names)
audit['archive']={'path':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'files':len(names),'CRC':'PASS'}
audit['package_validation_directory']=str(stage)
audit['totals']={k:sum(state()[str(n)][k] for n in range(1,10)) for k in ['pages','figures','tables']}
save(REPORTS/'formal_final_audit.json',audit)
with (REPORTS/'REPORT_PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n最终检查：九份独立编译通过，全页视觉检查通过；ZIP CRC 校验通过。\n'
            'Lab 4 Q2：已补齐色调控制实验。Lab1 原阈值失败在独立复跑中保留。\n')
print(json.dumps({'archive':audit['archive'],'totals':audit['totals'],'input_hashes':len(tracked)},ensure_ascii=False,indent=2),flush=True)
