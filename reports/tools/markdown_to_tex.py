"""Convert the measured Russian lab report into a self-contained six-section TeX source."""
import argparse
import re
from pathlib import Path


SYMBOLS={'→':r'\(\to\)','≤':r'\(\leq\)','≥':r'\(\geq\)','≈':r'\(\approx\)',
         'Δ':r'\(\Delta\)','θ':r'\(\theta\)','ε':r'\(\varepsilon\)','Θ':r'\(\Theta\)',
         '∥':r'\(\Vert\)','·':r'\(\cdot\)','×':r'\(\times\)','−':'-','–':'--','—':'---'}


def escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                  '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(SYMBOLS.get(c,replacements.get(c,c)) for c in text)


def inline(text):
    pieces=re.split(r'(`[^`]+`)',text)
    return ''.join(r'\path{'+p[1:-1]+'}' if p.startswith('`') else escape(p) for p in pieces)


def convert(markdown):
    lines=markdown.splitlines(); output=[]; i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('# '):
            i+=1; continue
        if line.startswith('## '):
            heading=re.sub(r'^\d+[.]\s*','',line[3:])
            output.append(r'\section{'+inline(heading)+'}')
            if 'Устройство harness' in heading:
                output.append(r'''\begin{align*}
\mathrm{EXP\_ID}&=\operatorname{SHA256}(C,\Theta,D,S)_{0:20},\\
\mathrm{RUN\_ID}&=\mathrm{EXP\_ID}\,\Vert\,\mathrm{UUID}.
\end{align*}''')
            if 'Результаты E0' in heading:
                output.append(r'''\begin{align*}
\overline m&=\frac1N\sum_{i=1}^{N}m_i,&
s_m&=\sqrt{\frac{\sum_i(m_i-\overline m)^2}{N-1}},\\
\mathrm{CI}_{95\%}&=\overline m\pm t_{0.975,N-1}\frac{s_m}{\sqrt N},&
\Delta_\theta&=\frac{\lVert\theta_A-\theta_B\rVert_2}{\lVert\theta_A\rVert_2+10^{-12}}.
\end{align*}''')
        elif line.startswith('### '):
            output.append(r'\subsection*{'+inline(line[4:])+'}')
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=lines[i]
                if not re.fullmatch(r'[\s|:\-]+',row):
                    table.append([c.strip() for c in row.strip('|').split('|')])
                i+=1
            count=len(table[0])
            col=(r'>{\raggedright\arraybackslash}p{\dimexpr\linewidth/'+str(count)+r'-2\tabcolsep-2\arrayrulewidth\relax}')
            output.append(r'\begingroup\scriptsize\setlength{\tabcolsep}{2pt}')
            output.append(r'\begin{longtable}{|'+('|'.join([col]*count))+r'|}\hline')
            for number,row in enumerate(table):
                output.append(' & '.join(inline(c) for c in row)+r' \\\hline')
                if number==0: output.append(r'\endhead')
            output.append(r'\end{longtable}\endgroup')
            continue
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
            if match:
                output.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{'+match[2]+r'}\caption{'+inline(match[1])+r'}\end{figure}')
        elif line.strip():
            output.append(inline(line)+r'\par')
        else:
            output.append('')
        i+=1
    return r'''\documentclass[11pt,a4paper]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage[margin=19mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,array,longtable,xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\emergencystretch=3em
\title{Лабораторная работа 0\\Инфраструктура воспроизводимых экспериментов}
\author{Ван Чжаофу\\ТПУ, курс «Нейронные сети»}
\date{2026}
\begin{document}
\maketitle
'''+ '\n'.join(output)+r'\end{document}'+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('markdown',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(convert(a.markdown.read_text(encoding='utf-8')),encoding='utf-8')
    print(a.out)


if __name__=='__main__': main()
