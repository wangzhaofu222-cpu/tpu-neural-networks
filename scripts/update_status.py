"""Refresh phase status from real experiment, build, and compatibility evidence."""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'reports/tools/update_stage_status.py'),run_name='__main__')
