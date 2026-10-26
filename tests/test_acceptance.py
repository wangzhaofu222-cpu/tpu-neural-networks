import pytest
from src.analysis.lab0_acceptance import read_run
from src.engine.runner import ExperimentRunner


def test_fixture_cannot_be_accepted_as_course_experiment(fixture_config):
    fixture_config['train']['epochs']=1
    run=ExperimentRunner(fixture_config).run()
    with pytest.raises(ValueError,match='real formal'):
        read_run(run)
