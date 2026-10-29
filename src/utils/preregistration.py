"""Verify a real immutable Git hypothesis blob before launching a formal comparison."""
import subprocess
from pathlib import Path

def verify_git_hypothesis(commit,path):
    root=Path('.').resolve();args=['git','-c','safe.directory='+root.as_posix()]
    if not commit:raise ValueError('Missing actual preregistration commit')
    ancestor=subprocess.run(args+['merge-base','--is-ancestor',commit,'HEAD'],capture_output=True)
    if ancestor.returncode:raise ValueError('Preregistration is not an ancestor of HEAD')
    blob=subprocess.run(args+['show',commit+':'+Path(path).as_posix()],capture_output=True)
    if blob.returncode:raise ValueError('Hypothesis absent from preregistration commit')
    normalize=lambda b:b.replace(b'\r\n',b'\n')
    if normalize(blob.stdout)!=normalize(Path(path).read_bytes()):raise ValueError('Preregistered hypothesis was modified')
    return True
