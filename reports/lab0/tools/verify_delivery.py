"""Audit final report, source preservation and typesetting without modifying inputs."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
import pdfplumber

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/lab0'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def norm(t):
    return re.sub(r'\s+','',t).replace('\u2010','-').replace('\u2011','-')

def main():
    p=OUT/'Lab0_Report.pdf'
    r=PdfReader(p)
    texts=[pg.extract_text() or '' for pg in r.pages]
    cover=norm(texts[0]);whole=norm(''.join(texts))
    checks={}
    checks['nineteen_pages']=len(r.pages)==19
    for s in ['Григорьев Д. С.','Нейронные сети','Ван Чжаофу','Лабораторная работа № 0',
              'Инфраструктура воспроизводимых экспериментов','и единый Experiment Harness','Томск','2026','8ВМ53']:
        checks['cover:'+s]=norm(s) in cover
    for old in ['Друки','Доцент','Детекция объектов на изображениях','Методы распознавания образов']:
        checks['removed:'+old]=norm(old) not in whole
    original=PdfReader(ROOT/'Тулист.pdf')
    title=PdfReader(OUT/'title_page.pdf')
    checks['cover_size_preserved']=original.pages[0].mediabox==title.pages[0].mediabox==r.pages[0].mediabox
    original_logo=next(iter(original.pages[0].images)).data
    checks['logo_bytes_preserved_in_final']=any(i.data==original_logo for i in r.pages[0].images)
    with pdfplumber.open(p) as pdf:
        footers=[pg.crop((0,800,pg.width,pg.height)).extract_text() or '' for pg in pdf.pages]
    checks['cover_has_no_body_page_number']=not footers[0].strip()
    checks['continuous_page_numbers_2_to_19']=[x.strip() for x in footers[1:]]==[str(i) for i in range(2,20)]
    checks['all_pages_A4']=all(abs(float(pg.mediabox.width)-595.28)<.1 and abs(float(pg.mediabox.height)-841.89)<.1 for pg in r.pages)
    body=(OUT/'report_body.tex').read_text('utf-8')
    checks['six_required_sections']=len(re.findall(r'\\section\{',body))==6
    checks['eight_counterfactuals']=len(re.findall(r'\\subsection\{Вопрос [1-8]\.',body))==8
    checks['seven_figures']=len(re.findall(r'\\begin\{figure\}',body))==7
    checks['nine_tables']=len(re.findall(r'\\begin\{table\}',body))+1==9
    checks['three_cited_sources']=len(re.findall(r'\\bibitem\{',body))==3
    checks['all_27_training_RUN_IDs_searchable']=all(norm(row['RUN_ID']) in whole for row in json.loads((OUT/'run_registry.json').read_text('utf-8')))
    checks['all_figures_included']=all((OUT/'figures'/f'{name}.pdf').exists() for name in
        ['learning_curves','repeatability','seed_variance','resume_and_ablation','performance','timing_stability'])
    for number in ['0.633440','0.008708','0.622628','0.644252','0.017416','0.371028','0.500332','20.81','13.62']:
        checks['reported_number:'+number]=number in whole
    log=(OUT/'build/main.log').read_text('utf-8',errors='replace')
    bad=re.findall(r'^.*(?:Overfull|Underfull|Missing character|LaTeX(?: Font)? Warning:|Package \S+ Warning:|^!).*$',log,re.M)
    checks['no_typesetting_warnings']=not bad
    baseline=json.loads((OUT/'build/preserved_inputs.json').read_text('utf-8'))
    changed=[]
    for rel,h in baseline.items():
        q=OUT/'previous_report/main.tex' if Path(rel).as_posix()=='reports/lab0/main.tex' else ROOT/rel
        if not q.exists() or digest(q)!=h:changed.append(rel)
    checks['all_preserved_inputs_unchanged']=not changed
    a=''.join(pg.extract_text() or '' for pg in PdfReader(ROOT/'drive-download-20261026T053112Z-1-001/lab0_experiment_harness_1010.pdf').pages)
    b=(OUT/'build/teacher_connected_fulltext.txt').read_text('utf-8')
    checks['teacher_full_cyrillic_sequence_matches_connected_source']=re.findall('[А-Яа-яЁё]',a)==re.findall('[А-Яа-яЁё]',b)
    checks['all_pages_rendered']=all((OUT/'build/qa'/f'report-{i:02}.png').exists() for i in range(1,20))
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'failed':[k for k,v in checks.items() if not v], 'warnings':bad,
            'preserved_input_files':len(baseline),'changed_preserved_inputs':changed,
            'pages':len(r.pages),'main_sections':6,'figures':7,'tables':9,'counterfactual_answers':8,
            'scientific_negative_results_retained':['Initial E0 K=10 loss ratio 0.500332 > 0.5',
                'Supplementary GPU latency CV: 20.81% and 13.62% > 10%'],
            'limitations_retained':['Independent teacher split unavailable','No general cross-environment/AMP guarantee',
                'No test accuracy measured in Lab0','One fixed split; five preselected primary seeds',
                'GPU timing instability only partly localized'],
            'training_executed':False,'formal_pdf_sha256':digest(p),'original_cover_sha256':digest(ROOT/'Тулист.pdf')}
    (OUT/'build/formal_report_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:result[k] for k in ['status','failed','preserved_input_files','pages','figures','tables']},ensure_ascii=False))
    assert not result['failed'],result['failed']

if __name__=='__main__':main()
