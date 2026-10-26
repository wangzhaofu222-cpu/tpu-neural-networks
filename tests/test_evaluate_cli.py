import json
import subprocess
import sys
from src.engine.runner import ExperimentRunner


def test_independent_evaluation_matches_summary(fixture_config):
    fixture_config['train']['epochs']=1
    path=ExperimentRunner(fixture_config).run()
    result=subprocess.run([sys.executable,'-X','utf8','-m','src.evaluate','--run',str(path)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr
    evaluation=json.loads(result.stdout.strip().splitlines()[-1])
    summary=json.loads((path/'summary.json').read_text())
    assert evaluation['accuracy']==summary['validation_metric']
