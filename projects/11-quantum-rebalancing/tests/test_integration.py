from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from benchmarks.run import run_benchmarks
from rebalancing.data import generate, load_config, load_portfolio, prepare
from rebalancing.engine import plan
from rebalancing.reporting import confidence, write_run_report
from rebalancing.solvers import Budget
from rebalancing.storage import RunStore


def config_file(tmp_path, **overrides):
    config={'schema_version':1,'dataset':'data/snapshot.json','runs':'runs',
            'generator':{'seed':7,'assets':2,'max_trade_lots':1},
            'budget':{'max_evaluations':8,'shots':64,'max_qubits':8},
            'benchmark':{'seeds':[7,19], 'asset_counts':[2], 'solvers':['classical','quantum-sim']}, **overrides}
    path=tmp_path/'demo.yaml'
    path.write_text(yaml.safe_dump(config))
    return path


def test_data_are_repeatable_immutable_and_hashed(tmp_path):
    config=load_config(config_file(tmp_path))
    first=prepare(config)
    assert prepare(config)==first
    assert load_portfolio(config['dataset'])==generate(7,2,1)
    with pytest.raises(ValueError,match='immutable'):
        prepare({**config,'generator':{'seed':8,'assets':2}})
    with Path(config['dataset']).open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError,match='SHA-256'):
        load_portfolio(config['dataset'])


def test_audit_content_ids_and_tamper_detection(tmp_path,tiny):
    store=RunStore(tmp_path)
    result=plan(tiny)
    run_id=store.save(result)
    assert store.save(result)==run_id
    assert store.get(run_id)==json.loads(json.dumps(result))
    assert store.verify()['entries_checked']==1
    second=plan(tiny)
    assert second['recommendation_id']==result['recommendation_id']
    store.save(second)
    assert store.verify()['valid']
    with sqlite3.connect(store.database) as connection:
        connection.execute('UPDATE runs SET previous_hash=? WHERE sequence=2',('corrupted',))
    assert not store.verify()['valid']


def test_api_feasible_validation_and_lookup(tmp_path,tiny):
    client=TestClient(create_app(tmp_path))
    assert client.get('/health').json()['execution']=='simulated_only'
    response=client.post('/rebalance',json={'portfolio':tiny.to_dict()})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['report']['feasible']
    assert client.get('/runs/'+result['run_id']).json()['recommendation_id']==result['recommendation_id']
    assert client.get('/runs/unknown').status_code==404
    assert client.get('/audit/verify').json()['valid']
    assert client.post('/rebalance',json={'portfolio':{},'unexpected':1}).status_code==422
    bad=tiny.to_dict()
    bad['assets'][0]['price_cents']=-1
    assert client.post('/rebalance',json={'portfolio':bad}).status_code==422
    assert client.post('/rebalance',json={'portfolio':tiny.to_dict(),'budget':{'shots':0}}).status_code==422
    assert client.post('/rebalance',json={'portfolio':replace(tiny,cash_reserve_cents=23000).to_dict()}).status_code==422


def test_api_resource_refusal_and_fallback(tmp_path,tiny):
    client=TestClient(create_app(tmp_path))
    data={'portfolio':tiny.to_dict(),'solver':'quantum-sim','budget':{'max_qubits':1}}
    refused=client.post('/rebalance',json=data)
    assert refused.status_code==422 and refused.json()['detail']['kind']=='resource_limit'
    accepted=client.post('/rebalance',json={**data,'fallback':'classical'})
    assert accepted.status_code==200
    assert accepted.json()['solver_used']=='classical'


def test_report_escapes_untrusted_symbols(tmp_path,tiny):
    portfolio=replace(tiny,assets=(replace(tiny.assets[0],symbol='<script>bad()</script>'),tiny.assets[1]))
    store=RunStore(tmp_path/'runs')
    store.save(plan(portfolio))
    path=write_run_report(tmp_path/'runs',tmp_path/'report.html')
    content=path.read_text()
    assert '&lt;script&gt;bad()&lt;/script&gt;' in content
    assert '<script>bad()</script>' not in content
    assert 'Post-NAV wt' in content


def test_cli_contract_from_another_working_directory(tmp_path):
    config=config_file(tmp_path)
    root=Path(__file__).resolve().parents[1]
    import os
    env={**os.environ,'PYTHONPATH':str(root)}
    for module,args in [
        ('project.prepare_data',['--config',str(config)]),
        ('project.run',['--config',str(config),'--solver','classical']),
        ('project.run',['--config',str(config),'--solver','quantum-sim']),
        ('project.report',['--runs',str(tmp_path/'runs'),'--output',str(tmp_path/'report.html')])]:
        proc=subprocess.run([sys.executable,'-m',module,*args],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
        assert proc.returncode==0,proc.stderr
    assert (tmp_path/'report.html').exists()
    assert RunStore(tmp_path/'runs').verify()['entries_checked']==2


def test_benchmark_retains_failures_and_declares_seeds(tmp_path):
    path=config_file(tmp_path,budget={'max_evaluations':4,'shots':32,'max_qubits':5})
    output=tmp_path/'bench'
    rows=run_benchmarks(str(path),output)
    assert len(rows)==4
    assert all(r['absolute_gap']==0 for r in rows if r['solver']=='classical')
    assert all(not r['success'] and 'ResourceLimitError' in r['error'] for r in rows if r['solver']=='quantum-sim')
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['seeds']==[7,19] and manifest['fallback'] is None
    assert (output/'comparison.html').exists() and (output/'comparison.summary.csv').exists()
    assert json.loads((output/'audit-verification.json').read_text())['valid']


def test_confidence_interval_requires_repeats():
    assert confidence([])==(None,None,None)
    assert confidence([3])==(3,None,None)
    avg,lo,hi=confidence([1,2,3])
    assert lo<avg<hi and avg==2


def test_audit_detects_invalid_json(tmp_path,tiny):
    store=RunStore(tmp_path)
    store.save(plan(tiny))
    with store.connect() as connection:
        connection.execute("UPDATE runs SET payload='invalid'")
    assert not store.verify()['valid']


def test_concurrent_idempotent_audit_exports(tmp_path,tiny):
    from concurrent.futures import ThreadPoolExecutor
    store=RunStore(tmp_path)
    result=plan(tiny)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids=list(pool.map(lambda _:store.save(result),range(8)))
    assert len(set(ids))==1
    assert store.verify()['entries_checked']==1
    assert json.loads((tmp_path/f'{ids[0]}.json').read_text())['recommendation_id']==result['recommendation_id']

