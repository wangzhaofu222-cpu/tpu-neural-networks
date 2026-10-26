"""Build a page-addressed requirement register from the preserved PDF extracts."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'requirements'
rows = []
inventory = []
for lab in range(10):
    pdf = next((ROOT / 'drive-download-20261026T053112Z-1-001').glob(f'lab{lab}_*.pdf'))
    txt = OUT / 'sources' / (pdf.stem + '.txt')
    raw = txt.read_text(encoding='utf-8')
    inventory.append({'lab': lab, 'path': str(pdf.relative_to(ROOT)), 'bytes': pdf.stat().st_size,
                      'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                      'pages': raw.count('=== PDF PAGE')})
    section = '0'
    serial = 0
    chunks = []
    # Preserve every paragraph, including acceptance bullets, boundary studies and questions.
    for page, content in re.findall(r'=== PDF PAGE (\d+) ===\n(.*?)(?=\n=== PDF PAGE|\Z)', raw, re.S):
        content = re.sub(r'(?m)^\s*\d+\s*$', '', content)
        for paragraph in re.split(r'\n\s*\n|\n(?=\s*(?:[•]|\d+[.]\s*[^\d]))', content):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue
            heading = re.match(r'^(\d+(?:\.\d+)?)\s{2,}(\S.*)', paragraph.strip())
            if heading:
                section = heading.group(1)
            serial += 1
            rid = f'L{lab}-R{serial:03d}'
            value = ' '.join(lines)
            optional = (lab == 0 and section.startswith('29')) or (lab in [1,2,3,4,5,7,8,9] and section.startswith('14')) or (lab == 6 and section.startswith('15'))
            rows.append({'id': rid, 'lab': lab, 'page': int(page), 'section': section,
                         'scope': 'OPTIONAL' if optional else 'SOURCE_CLAUSE',
                         'text': value, 'status': 'NOT RUN', 'evidence': []})
            chunks.append(f'| {rid} | {page} | {section} | {"附加" if optional else "原文条款"} | {value.replace("|", "&#124;")} | NOT RUN | — |')
    (OUT / f'lab{lab}_clauses.md').write_text(
        f'# Lab {lab} 原文逐条登记\n\n来源：`{pdf.name}`。页码为 PDF 页码。\n\n'
        '此登记保留原文段落、项目、公式文本和脚注；公式须结合 PDF，提取文本不是公式权威版本。'
        '背景、理论及参考文献也保留以避免丢失上下文；SOURCE_CLAUSE 不等于全部必做。必做范围以主矩阵为准。'
        'NOT RUN 表示尚未以实现/测试/实验验证原文条款，不表示未阅读。\n\n'
        '| ID | 页 | 章 | 类型 | 条款原文 | 状态 | 证据 |\n|---|---:|---|---|---|---|---|\n' + '\n'.join(chunks) + '\n', encoding='utf-8')
(OUT / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'clause_register.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'{len(inventory)} PDFs, {sum(x["pages"] for x in inventory)} pages, {len(rows)} source clauses')
