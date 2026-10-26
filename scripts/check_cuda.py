import subprocess
import sys
from pathlib import Path

if __name__=='__main__':
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_environment.py'))],check=True)
