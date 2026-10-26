"""Capture the actual full package environment; does not invent a dependency lock."""
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
result=subprocess.run([sys.executable,'-m','pip','freeze','--all'],check=True,capture_output=True,text=True)
(root/'requirements.lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu126\n'+result.stdout,encoding='utf-8')
result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
print(result.stdout,result.stderr)
if result.returncode:
    raise SystemExit(result.returncode)
