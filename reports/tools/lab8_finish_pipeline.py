"""Continue ordered final stages after the separate E3 worker completes."""
import time,subprocess,sys
from pathlib import Path

def main():
    deadline=time.monotonic()+3600
    while not Path('experiments/lab8/E3.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('E3 completion not observed; inspect its real worker')
        time.sleep(2)
    phases=[['reports.tools.lab8_execute','selective'],['reports.tools.lab8_resolve_provenance'],['reports.tools.lab8_finalize'],['reports.tools.lab8_artifacts'],['reports.tools.lab8_regenerate']]
    for phase in phases:
        print('START',phase,flush=True);subprocess.run([sys.executable,'-X','utf8','-m',*phase],check=True)
    with open('tmp/lab8_all_final.log','w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8','-m','pytest','-q','--junitxml=docs/environment/tests-lab8-final.xml'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('READY_FOR_VISUAL_QA_AND_INDEPENDENT_AUDIT',flush=True)

if __name__=='__main__':main()
