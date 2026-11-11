from prepare import ROOT,OUT,TOOLS,source,sha,save
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,re,os,subprocess,shutil,sys
TEX=Path('D:/Tex Live/data/texlive/2026/bin/windows')
TITLES=['可复现实验基础设施与统一实验框架','深层网络的可训练性：初始化与残差连接','CNN与ViT：样本效率及分布偏移','注意力、位置表示与长序列外推','自监督表示学习与数据增强','图神经网络与图结构作用','能量模型与归一化流','扩散模型与精确Score诊断','模型可靠性、校准与不确定性','多模态表示与音视频情绪识别']
PREAMBLE=r'''% !TeX program = xelatex
\documentclass[UTF8,a4paper,zihao=-4,fontset=windows]{ctexart}
\usepackage[left=25mm,right=20mm,top=20mm,bottom=22mm]{geometry}
\setmainfont{Times New Roman}\setsansfont{Arial}\setmonofont{Consolas}
\usepackage{amsmath,amssymb,fix-cm,graphicx,booktabs,longtable,array,float,pdfpages}
\usepackage{setspace,caption,enumitem,needspace,etoolbox,tikz,xurl}
\usetikzlibrary{positioning}
\usepackage[unicode,hidelinks,bookmarksnumbered]{hyperref}
\urlstyle{same}\setstretch{1.2}\setlength{\parindent}{2em}
\sloppy\setlength{\emergencystretch}{4em}\setlength{\parskip}{0pt}
\captionsetup{font=small,labelsep=endash}\captionsetup[table]{position=top,skip=7pt}
\setcounter{tocdepth}{1}\raggedbottom
\widowpenalty=10000\clubpenalty=10000\displaywidowpenalty=10000
\AtBeginEnvironment{table}{\setstretch{1.05}}
\DeclareRobustCommand{\run}[1]{\hyperlink{#1}{\textnormal{#1}}}
\newcommand{\FitTable}[1]{\sbox0{#1}\ifdim\wd0>\linewidth\resizebox{\linewidth}{!}{\usebox0}\else\usebox0\fi}
'''

def translate():
    units=json.loads((TOOLS/'units.json').read_text('utf-8'));translations={}
    for p in TOOLS.glob('zh_*.tsv'):
        for line in p.read_text('utf-8').splitlines():
            if not line.strip():continue
            key,value=line.split('\t',1);assert key not in translations,key;translations[key]=value
    short={'равной ширины':'等宽','равное число':'等数量','Шум':'噪声','Размытие':'模糊','Яркость':'亮度','Контраст':'对比度','Закрытие':'遮挡'}
    for k,u in units.items():
        if k not in translations:
            s=u['prompt']
            for a,b in short.items():s=s.replace(a,b)
            if not re.search('[А-Яа-яЁё]',s):translations[k]=s
    missing=set(units)-set(translations);assert not missing,sorted(missing)
    files={};audit=[]
    for key,u in units.items():
        s=translations[key]
        assert Counter(re.findall(r'@@\d+@@',s))==Counter(re.findall(r'@@\d+@@',u['prompt'])),(key,'formula/reference placeholders differ')
        # Permit wrapping of prose lists without touching protected formulae or identifiers.
        s=re.sub(r'(?<=[0-9A-Za-z])/(?=[0-9A-Za-z])',r'/\\allowbreak ',s)
        for i,v in enumerate(u['protected']):s=s.replace(f'@@{i}@@',v.replace(r'\text{эп}',r'\text{每轮}'))
        assert '@@' not in s,key
        assert not re.search('[А-Яа-яЁё]',s),(key,s)
        index=(u['lab'],u['file'])
        if index not in files:
            files[index]=(source(u['lab'])/u['file']).read_text('utf-8-sig').splitlines()
        files[index][u['line']]=s
        audit.append({'unit':key,'source_file':u['file'],'source_line':u['line']+1,'translation':s})
    for (n,rel),lines in files.items():
        p=OUT/'04_源码与核查'/f'lab{n}_zh'/rel;p.write_text('\n'.join(lines)+'\n','utf-8')
    # All table files must exist, including purely numeric/English ones.
    for n in range(10):
        target=OUT/'04_源码与核查'/f'lab{n}_zh'
        for p in (source(n)/'tables').glob('*.tex'):
            q=target/'tables'/p.name
            if not q.exists():shutil.copy2(p,q)
        title=TITLES[n]
        cover='''\\begin{document}
\\begin{titlepage}\\centering
{\\Large 托木斯克理工大学\\par}\\vspace{1cm}
{\\large 《神经网络》课程\\par}\\vfill
{\\LARGE 实验 '''+str(n)+'''\\par}\\vspace{.6cm}
{\\Large '''+title+'''\\par}\\vspace{1cm}
{\\large 中文对照译本\\par}\\vspace{.4cm}
用于理解原报告与准备口头答辩\\par\\vfill
学生：Ван Чжаофу\\quad 班级：8ВМ53\\par
教师：Григорьев Д. С.\\par\\vspace{1cm}托木斯克 · 2026
\\end{titlepage}
\\tableofcontents\\clearpage
\\noindent\\textbf{阅读说明：}本译本对应所附俄语正式报告的全部章节、表格、公式和反事实回答，保留实测数值与完整运行标识。中文分页可能不同；请按章节、图表编号和 RUN\\_ID 对照。图内原始英文或俄文坐标标注保持，中文图题和正文解释其含义。本次只整理与翻译，不训练模型或改写实验结果。
\\par\\medskip
\\input{report_body.tex}
\\end{document}
'''
        (target/'main.tex').write_text(PREAMBLE+cover,'utf-8')
        (target/'build.ps1').write_text("$ErrorActionPreference='Stop'\nPush-Location -LiteralPath $PSScriptRoot\ntry { latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex; if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' } } finally { Pop-Location }\n",'utf-8-sig')
    save(OUT/'04_源码与核查'/'translation_alignment.json',audit)
    print('Translated units',len(audit),flush=True)

def compile_one(n):
    target=OUT/'04_源码与核查'/f'lab{n}_zh';env=os.environ.copy();env.update(PATH=str(TEX)+';'+env['PATH'],LC_ALL='C',LANG='C')
    p=subprocess.run([str(TEX/'latexmk.exe'),'-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir=build','main.tex'],cwd=target,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'console.log').write_bytes(p.stdout)
    if p.returncode:raise RuntimeError(f'Lab{n}: '+p.stdout.decode('utf-8','replace')[-2400:])
    shutil.copy2(target/'build/main.pdf',OUT/'02_中文对照译本'/f'Lab{n}_中文对照译本.pdf')
    return n

if __name__=='__main__':
    translate()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(compile_one,n) for n in range(10)]):print('Compiled Chinese Lab',f.result(),flush=True)
