from pathlib import Path
import re,shutil,subprocess,os
from prepare import OUT,TOOLS
from build import PREAMBLE,TEX

def escape(s):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)

def main():
    target=OUT/'04_源码与核查'/'guide';target.mkdir(exist_ok=True)
    md=(TOOLS/'答辩提纲.md').read_text('utf-8');blocks=md.split('\n\n');body=[]
    def paragraph(b):
        p=escape(b.replace('\n',' '))
        for prefix in ['中文：','中文答：','Русский:','Ответ:','Вопрос преподавателя:','Возможный вопрос:']:
            if b.startswith(prefix):p=r'\textbf{'+escape(prefix)+'}'+p[len(escape(prefix)):];break
        return p+'\n\\par\\medskip'
    question=[]
    def flush_question():
        if question:
            body.append(r'\par\noindent\begin{minipage}{\linewidth}'+'\n'+'\n'.join(question)+'\n'+r'\end{minipage}\par\medskip')
            question.clear()
    for b in blocks:
        b=b.strip()
        if not b:continue
        if b.startswith('#'):flush_question()
        if b.startswith('# '):
            if b.startswith('# 神经网络'):continue
            body.append(r'\clearpage\section{'+escape(b[2:])+'}')
        elif b.startswith('## '):body.append(r'\subsection*{'+escape(b[3:])+'}')
        elif b.startswith('### '):question.append(r'\subsubsection*{'+escape(b[4:])+'}')
        else:
            if question:question.append(paragraph(b))
            else:body.append(paragraph(b))
    flush_question()
    preamble=PREAMBLE.replace(r'\setcounter{tocdepth}{1}',r'\setcounter{tocdepth}{1}\setcounter{secnumdepth}{0}')
    preamble+=r'''
\usepackage{polyglossia}
\setdefaultlanguage{russian}\setotherlanguage{english}
\newfontfamily\russianfont{Times New Roman}
\newfontfamily\russianfontsf{Arial}\newfontfamily\russianfonttt{Consolas}
\setlength{\parindent}{0pt}
\ctexset{section={format=\Large\bfseries},subsection={format=\large\bfseries},subsubsection={format=\normalsize\bfseries}}
\hypersetup{pdftitle={Lab 0-9 中俄双语答辩指南},pdfauthor={Ван Чжаофу}}
'''
    cover=r'''
\begin{document}
\begin{titlepage}\centering
{\Large TPU · Нейронные сети\par}\vspace{1cm}
{\LARGE Lab 0–9 中俄双语答辩指南\par}\vspace{.7cm}
{\Large Пособие для защиты\par}\vspace{1cm}
理解实验 · 解释方法 · 陈述结果 · 回答追问\par
Цель · Метод · Результат · Вопросы\par\vfill
10个实验\quad 44组问答\quad 逐段中俄对应\par\vspace{.5cm}
学生：Ван Чжаофу\quad 班级：8ВМ53\par\vspace{1cm}
2026年11月11日
\end{titlepage}
\renewcommand{\contentsname}{目录 / Содержание}
\tableofcontents\clearpage
'''
    (target/'main.tex').write_text(preamble+cover+'\n\n'.join(body)+'\n\\end{document}\n','utf-8')
    env=os.environ.copy();env.update(PATH=str(TEX)+';'+env['PATH'],LC_ALL='C',LANG='C')
    p=subprocess.run([str(TEX/'latexmk.exe'),'-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir=build','main.tex'],cwd=target,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'console.log').write_bytes(p.stdout)
    if p.returncode:raise RuntimeError(p.stdout.decode('utf-8','replace')[-3000:])
    dst=OUT/'03_中俄答辩提纲'
    shutil.copy2(target/'build/main.pdf',dst/'Lab0-9_中俄双语答辩指南.pdf')
    shutil.copy2(TOOLS/'答辩提纲.md',dst/'Lab0-9_中俄双语答辩指南.md')
    shutil.copy2(OUT/'04_源码与核查/lab0_zh/build.ps1',target/'build.ps1')
    print('Guide compiled; questions:',len(re.findall(r'^### ',md,re.M)))

if __name__=='__main__':main()
