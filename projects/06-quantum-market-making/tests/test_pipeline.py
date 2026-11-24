import json
import subprocess
import sys
from market_making.experiment import run_experiment
from market_making.reporting import report, collect


def test_cli_contract_and_full_pipeline(config,tmp_path):
    runs,data=tmp_path/'runs',tmp_path/'data'
    config['seeds']=[11]
    config['scenarios']=['normal']
    config['data']['steps']=60
    for solver in ('fixed-spread','inventory-skew','avellaneda-stoikov','classical','quantum-sim'):
        path,manifest=run_experiment(config,solver,runs,data)
        assert manifest['status']=='complete'
        trial=path/'normal'/'11'
        assert (trial/'trajectory.csv').exists()
        assert (trial/'metrics.prom').exists()
        metrics=json.loads((trial/'metrics.json').read_text())
        assert metrics['optimizer_fallbacks']==0
        for record in json.loads((trial/'optimizer.json').read_text()):
            assert record['training_end_ns']<record['effective_timestamp_ns']
    html=report(runs,tmp_path/'report.html')
    assert 'QAOA' in html.read_text()
    assert '0 failed trials' in html.read_text()
    assert html.with_suffix('.csv').exists()
    for module in ('prepare_data','run','report'):
        process=subprocess.run([sys.executable,'-m',f'project.{module}','--help'],capture_output=True,text=True)
        assert process.returncode==0


def test_report_does_not_count_duplicate_seeds(config,tmp_path):
    config['seeds']=[11]
    config['scenarios']=['normal']
    config['data']['steps']=60
    for _ in range(2):
        run_experiment(config,'fixed-spread',tmp_path/'runs',tmp_path/'data')
    groups,failures,_=collect(tmp_path/'runs')
    assert len(groups)==1 and len(next(iter(groups.values())))==1
    assert not failures
