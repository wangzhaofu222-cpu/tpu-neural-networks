"""Portable PDF export from the same measured report text, with embedded Cyrillic fonts."""
import argparse
import html
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY,TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,LongTable,TableStyle,KeepTogether


def markup(text):
    text=text.replace('–','-').replace('—','-').replace('‑','-')
    parts=re.split(r'(`[^`]+`)',text)
    return ''.join('<font name="Code" size="7.8">'+html.escape(p[1:-1])+'</font>' if p.startswith('`') else html.escape(p) for p in parts)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('markdown',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for name,file in [('Body','times.ttf'),('BodyBold','timesbd.ttf'),('Code','cour.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportBody',fontName='Body',fontSize=10.3,leading=13.8,spaceAfter=6,alignment=TA_JUSTIFY,splitLongWords=True))
    styles.add(ParagraphStyle('ReportTitle',fontName='BodyBold',fontSize=17,leading=21,spaceAfter=14))
    styles.add(ParagraphStyle('ReportH2',fontName='BodyBold',fontSize=13,leading=16,spaceBefore=12,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle('ReportH3',fontName='BodyBold',fontSize=11,leading=14,spaceBefore=8,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle('RegistryH3',parent=styles['ReportH3'],keepWithNext=False))
    styles.add(ParagraphStyle('Cell',fontName='Body',fontSize=7.8,leading=9.5,alignment=TA_LEFT,splitLongWords=True))
    styles.add(ParagraphStyle('CaptionRU',fontName='Body',fontSize=9,leading=11,spaceAfter=9))
    width=A4[0]-36*mm
    elements=[]; lines=a.markdown.read_text(encoding='utf-8').splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip()
        if line.startswith('# '): elements.append(Paragraph(markup(line[2:]),styles['ReportTitle']))
        elif line.startswith('## '): elements.append(Paragraph(markup(line[3:]),styles['ReportH2']))
        elif line.startswith('### '): elements.append(Paragraph(markup(line[4:]),styles['RegistryH3'] if 'Реестр' in line else styles['ReportH3']))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                item=lines[i].strip()
                if not re.fullmatch(r'[\s|:\-]+',item): rows.append([c.strip() for c in item.strip('|').split('|')])
                i+=1
            header=rows[0]; count=len(header)
            weights=[1/count]*count
            if 'RUN_ID' in header and count==6: weights=[.07,.16,.34,.12,.11,.20]
            if count==8: weights=[.06,.16,.12,.10,.07,.17,.13,.09]
            rows=[[Paragraph(markup(c),styles['Cell']) for c in row] for row in rows]
            table=LongTable(rows,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7ecf1')),
                ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#aab5bf')),('LEFTPADDING',(0,0),(-1,-1),3),
                ('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            elements.extend([table,Spacer(1,6)])
            continue
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
            image=Image(str(a.markdown.parent/match[2]))
            image.drawHeight=image.imageHeight*width/image.imageWidth
            image.drawWidth=width
            elements.append(KeepTogether([image,Paragraph(markup(match[1]),styles['CaptionRU'])]))
        elif line:
            elements.append(Paragraph(markup(line),styles['ReportBody']))
        i+=1
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#b9c2ca')); canvas.line(18*mm,15*mm,A4[0]-18*mm,15*mm)
        canvas.setFont('Body',8); canvas.setFillColor(colors.HexColor('#465362'))
        canvas.drawString(18*mm,11*mm,'ТПУ · ЛР0 · Ван Чжаофу')
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
    doc=SimpleDocTemplate(str(a.out),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=21*mm,
                         title='Лабораторная работа 0. Воспроизводимые эксперименты',author='Ван Чжаофу')
    doc.build(elements,onFirstPage=footer,onLaterPages=footer)
    print(a.out)


if __name__=='__main__': main()
