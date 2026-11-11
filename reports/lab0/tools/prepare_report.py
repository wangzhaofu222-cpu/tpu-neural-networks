"""Prepare report-only assets; never train or rewrite experimental evidence."""
from pathlib import Path
import hashlib
import json
import shutil
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/lab0'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    for name in ['assets', 'figures', 'tables', 'build/qa', 'previous_report']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    old = OUT / 'previous_report/main.tex'
    if not old.exists():
        shutil.copy2(OUT / 'main.tex', old)
    # Existing report.pdf and report.ru.md are retained at their original paths.
    baseline = OUT / 'build/preserved_inputs.json'
    if not baseline.exists():
        paths = [ROOT / 'Тулист.pdf', *ROOT.glob('drive-download-*/*.pdf')]
        for d in ['src', 'tests', 'configs', 'splits', 'experiments']:
            paths.extend(p for p in (ROOT / d).rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts)
        # All original Lab0 logs, checkpoints, diagnostics, and existing report evidence.
        runs = json.loads((OUT / 'run_registry.json').read_text('utf-8'))
        for row in runs:
            paths.extend(p for p in (ROOT / 'runs' / row['RUN_ID']).rglob('*') if p.is_file())
        paths.extend(p for p in OUT.rglob('*') if p.is_file()
                     and p.parts[len(OUT.parts)] not in
                     ['tools', 'build', 'previous_report', 'assets', 'figures', 'tables'])
        paths = sorted(set(paths))
        baseline.write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in paths},
                                       ensure_ascii=False, indent=2), 'utf-8')
    source = ROOT / 'Тулист.pdf'
    r = PdfReader(source)
    assert len(r.pages) == 1
    p = r.pages[0]
    c = ContentStream(p.get_contents(), r)
    # Suppress text-showing operators only in the old report-title block and
    # the outdated teacher rank/name. Retain all other text, the original logo,
    # every graphic, signature rules, transforms and page dimensions.
    filtered = []
    suppress = False
    removed = 0
    for args, op in c.operations:
        if op == b'BT':
            suppress = False
        if op == b'Tm':
            x, y = float(args[4]), float(args[5])
            suppress = (350 < y < 410) or (600 < y < 612 and (x < 220 or x > 400))
        if suppress and op in [b'Tj', b'TJ', b"'", b'"']:
            removed += 1
        else:
            filtered.append((args, op))
        if op == b'ET':
            suppress = False
    c.operations = filtered
    p[NameObject('/Contents')] = c
    w = PdfWriter()
    w.add_page(p)
    base = OUT / 'assets/title_base.pdf'
    w.write(base)
    text = PdfReader(base).pages[0].extract_text()
    assert all(s not in text for s in ['Друки', 'Доцент', 'Детекция', 'распознавания'])
    assert 'Ван Чжаофу' in text and '8ВМ53' in text
    assert list(PdfReader(source).pages[0].images)[0].data == list(PdfReader(base).pages[0].images)[0].data
    teacher = ROOT / 'drive-download-20261026T053112Z-1-001/lab0_experiment_harness_1010.pdf'
    tr = PdfReader(teacher)
    assert len(tr.pages) == 24
    (OUT / 'build/teacher_original_fulltext.txt').write_text(
        '\n\n'.join(f'=== PAGE {i+1} ===\n{p.extract_text()}' for i, p in enumerate(tr.pages)), 'utf-8')
    (OUT / 'build/cover_source_check.json').write_text(json.dumps({
        'original': str(source), 'sha256': digest(source), 'pages': len(r.pages),
        'page_size_bp': list(map(float, p.mediabox)), 'removed_text_show_operators': removed,
        'original_logo_bytes_preserved': True, 'group_source': 'User-supplied original cover: 8ВМ53',
        'method': 'Remove old text-showing operators, retain original page, overlay new text with XeLaTeX',
        'teacher_pdf_sha256': digest(teacher), 'teacher_pdf_pages': len(tr.pages),
    }, ensure_ascii=False, indent=2), 'utf-8')
    print('Prepared cover base and read-only evidence baseline.')

if __name__ == '__main__':
    main()
