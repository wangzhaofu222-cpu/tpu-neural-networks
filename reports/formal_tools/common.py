"""Report-only tools. No training, evaluation, or modification of experiment files."""
from pathlib import Path
import json, hashlib, re, shutil, subprocess, os
ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'reports'
STATE=REPORTS/'formal_report_state.json'
TEX=Path('D:/Tex Live/data/texlive/2026/bin/windows')

def load(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state():return load(STATE) if STATE.exists() else {str(n):{'status':'NOT_STARTED'} for n in range(1,10)}
def update(n,**kw):
    d=state();d[str(n)].update(kw);save(STATE,d)
    text='# 正式报告制作进度\n\n报告依据原始实验与 2026-11-06 补充实验更新。历史记录和上一版报告保留；源码位于各 Lab 的 formal 子目录。\n\n'
    text+='| Lab | 实验资料读取 | 报告完成 | PDF 编译 | 视觉检查 | 状态 |\n|---|---|---|---|---|---|\n'
    for i in range(1,10):
        r=d[str(i)];text+='| Lab '+str(i)+' | '+' | '.join(str(r.get(k,'')) for k in ['read','written','compiled','visual','status'])+' |\n'
    text+='\n说明：Lab 6 按原始讲义撰写 EBM 与 Flow，不采用提示中的 GAN 线索。报告完成状态与科学指标是否达到阈值分别记录。\n'
    (REPORTS/'REPORT_PROGRESS.md').write_text(text,'utf-8')

def track(paths):
    p=REPORTS/'formal_inputs_sha256.json';d=load(p) if p.exists() else {}
    for q in paths:
        q=Path(q);key=q.relative_to(ROOT).as_posix();h=sha(q)
        if key in d:assert d[key]==h,key
        d[key]=h
    save(p,d)

def setup(n,title_lines):
    out=REPORTS/f'lab{n}'/'formal';out.mkdir(parents=True,exist_ok=True)
    for x in ['assets','figures','tables','build/qa']:(out/x).mkdir(parents=True,exist_ok=True)
    src=REPORTS/'lab0'
    track([src/'main.tex',src/'title_page.tex',src/'Lab0_Report.pdf',ROOT/'Тулист.pdf',src/'assets/title_base.pdf'])
    shutil.copy2(src/'assets/title_base.pdf',out/'assets/title_base.pdf')
    pre=(src/'main.tex').read_text('utf-8')
    pre=re.sub(r'\\hypersetup\{.*?\}\n',lambda m:r'\hypersetup{pdftitle={Лабораторная работа '+str(n)+r'},pdfauthor={Ван Чжаофу},pdfsubject={Нейронные сети}}'+'\n',pre)
    pre=pre.replace('\\setcounter{tocdepth}{2}','\\setcounter{tocdepth}{1}')
    (out/'main.tex').write_text(pre,'utf-8')
    cover=(src/'title_page.tex').read_text('utf-8')
    start=cover.index('\\CoverText{319.5}{362.33}')
    end=cover.index('\\CoverText{477.0}',start)
    new='\\CoverText{319.5}{362.33}{14.05}{Лабораторная работа № '+str(n)+'}\n'
    for i,line in enumerate(title_lines):
        q=('«' if i==0 else '')+line+('»' if i==len(title_lines)-1 else '')
        new+='\\CoverText{319.5}{'+str(383+17*i)+'}{13.5}{'+q+'}\n'
    new+='\\CoverText{319.5}{'+str(406+17*len(title_lines))+'}{14.05}{по дисциплине «Нейронные сети»}\n'
    cover=cover[:start]+new+cover[end:]
    cover=cover.replace('\\newcommand{\\LabNumber}{0}','\\newcommand{\\LabNumber}{'+str(n)+'}')
    # Unused Lab0 title commands must not remain in the reusable source.
    cover=re.sub(r'\\newcommand\{\\LabTitleFirst\}\{[^\n]*\}',r'\\newcommand{\\LabTitleFirst}{'+title_lines[0]+'}',cover)
    cover=re.sub(r'\\newcommand\{\\LabTitleSecond\}\{[^\n]*\}',r'\\newcommand{\\LabTitleSecond}{'+title_lines[-1]+'}',cover)
    (out/'title_page.tex').write_text(cover,'utf-8')
    script=(src/'build.ps1').read_text('utf-8').replace('Lab0_Report.pdf',f'Lab{n}_Report.pdf')
    (out/'build.ps1').write_text(script,'utf-8-sig')
    update(n,status='IN_PROGRESS',read='PASS')
    return out

def table(out,name,caption,heads,rows,spec=None,size='small'):
    spec=spec or 'l'+'r'*(len(heads)-1)
    t='\\begin{table}[H]\\centering\\'+size+'\n\\setlength{\\tabcolsep}{4pt}\n\\caption{'+caption+'}\\label{tab:'+name+'}\n'
    t+='\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(heads)+' \\\\\n\\midrule\n'
    t+=''.join(' & '.join(str(v) for v in row)+' \\\\\n' for row in rows)
    t+='\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (out/'tables'/f'{name}.tex').write_text(t,'utf-8')
    return '\\input{tables/'+name+'.tex}\n'

def figure(out,source,name,caption,height='.42'):
    source=Path(source);track([source]);dest=out/'figures'/(name+source.suffix);shutil.copy2(source,dest)
    return '\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth,height='+height+'\\textheight,keepaspectratio]{figures/'+dest.name+'}\n\\caption{'+caption+'}\\label{fig:'+name+'}\n\\end{figure}\n'

def registry(out,rows):
    text='\\clearpage\n\\section*{Приложение. Идентификация запусков}\n\\addcontentsline{toc}{section}{Приложение. Идентификация запусков}\n'
    text+='\\begingroup\\small\\setstretch{1.05}\n\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{3.5cm}p{10.8cm}@{}}\n'
    text+='\\caption{Полные RUN\\_ID источников результатов.}\\\\\n\\toprule Код & Условие & RUN\\_ID \\\\ \\midrule \\endhead\n'
    for alias,condition,rid in rows:
        condition=condition.replace('_',' ')
        text+='\\hypertarget{'+alias+'}{'+alias+'} & '+condition+' & {\\fontsize{8.8}{11}\\selectfont\\texttt{'+rid+'}} \\\\\n'
    return text+'\\bottomrule\\end{longtable}\\endgroup\n'

def bibliography(n,title,extra=''):
    return '\\clearpage\n\\renewcommand{\\refname}{Список использованных источников}\n\\begin{thebibliography}{9}\n\\addcontentsline{toc}{section}{Список использованных источников}\n\\bibitem{teacher} Лабораторная работа '+str(n)+'. '+title+'. Курс «Нейронные сети». Методические указания, 2026.\n'+extra+'\\end{thebibliography}\n'

def compile_report(n):
    out=REPORTS/f'lab{n}'/'formal';env=os.environ.copy();env.update(PATH=str(TEX)+';'+env['PATH'],LC_ALL='C',LANG='C')
    for stem in ['title_page','main']:
        result=subprocess.run([str(TEX/'latexmk.exe'),'-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir=build',stem+'.tex'],cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/'build'/f'{stem}-console.log').write_bytes(result.stdout)
        if result.returncode:raise RuntimeError(result.stdout.decode('utf-8','replace')[-4000:])
        if stem=='title_page':shutil.copy2(out/'build/title_page.pdf',out/'title_page.pdf')
    shutil.copy2(out/'build/main.pdf',out/f'Lab{n}_Report.pdf')
    shutil.copy2(out/f'Lab{n}_Report.pdf',REPORTS/f'lab{n}'/f'Lab{n}_Report.pdf')
    update(n,written='PASS',compiled='PASS')
    return out

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:compile_report(int(sys.argv[1]))
    else:update(1)
