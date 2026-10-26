"""Audit invariants; does not claim experimental requirements have passed."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
directory=root/'docs/requirements'
inventory=json.loads((directory/'source_inventory.json').read_text(encoding='utf-8'))
clauses=json.loads((directory/'clause_register.json').read_text(encoding='utf-8'))
assert len(inventory)==10
assert sum(p['pages'] for p in inventory)==152
assert len({row['id'] for row in clauses})==len(clauses)
for item in inventory:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
    lab=item['lab']
    assert {r['page'] for r in clauses if r['lab']==lab}==set(range(1,item['pages']+1))
    assert (directory/f'lab{lab}_clauses.md').is_file()
for name in ['course_overview','requirements_matrix','dependencies','missing_resources','implementation_plan']:
    assert (directory/f'{name}.md').is_file()
print(f'PASS: 10 immutable PDFs, 152 pages, {len(clauses)} unique source clause IDs, all pages represented, five required documents.')
