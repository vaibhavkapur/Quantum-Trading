import copy
import json
import numpy as np
import pytest
from market_making.config import validate, load_config
from market_making.data import generate, prepare, load_dataset, validate_frames
from market_making.policy import baseline
from market_making.simulation import simulate, policy_cost


def test_data_deterministic_immutable_and_verified(config,tmp_path):
    path=prepare(config,tmp_path)
    assert prepare(config,tmp_path)==path
    manifest,frames=load_dataset(path)
    assert frames[('normal',11)]==generate(config,11,'normal')
    assert manifest['splits']=={'train':[0,90],'validation':[90,126],'test':[126,180]}
    file=path/manifest['files'][0]['file']
    file.write_bytes(file.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='checksum'):
        prepare(config,tmp_path)


def test_stress_begins_only_in_test(config):
    normal=generate(config,11,'normal')
    volatile=generate(config,11,'volatile')
    assert normal[:126]==volatile[:126]
    assert normal[126:]!=volatile[126:]


@pytest.mark.parametrize('change',[
    lambda f: f[1].update(timestamp_ns=f[0]['timestamp_ns']),
    lambda f: f[0].update(mid_ticks=float('nan')),
    lambda f: f[0].update(market_buy_qty=-1),
    lambda f: f[0].update(bids=[[10001,2]],asks=[[10000,2]]),
    lambda f: f[0].update(bids=[[9999,2],[9999,1]]),
])
def test_bad_replays_rejected(config,change):
    frames=generate(config,11,'normal')
    change(frames)
    with pytest.raises(ValueError):
        validate_frames(frames)


def test_deterministic_regression_and_accounting(config):
    frames=generate(config,11,'normal')
    p=baseline('inventory-skew',config)
    a,b=simulate(frames,config,p),simulate(frames,config,p)
    for key in a['metrics']:
        if not key.endswith(('_seconds','_ns')):
            assert a['metrics'][key]==b['metrics'][key]
    m=a['metrics']
    assert m['net_pnl']==pytest.approx(.960106,abs=1e-8)
    cash=-sum(f['side']*f['price']*.01*f['quantity']+f['fee'] for f in a['fills'])-m['cancellation_cost']
    assert m['net_pnl']==pytest.approx(cash+m['residual_inventory']*frames[-1]['mid_ticks']*.01)
    assert m['max_abs_inventory']<=config['execution']['max_inventory']
    assert 0<=m['fill_probability']<=1
    assert m['residual_inventory']==0


def test_objective_analytic(config):
    result={'metrics':dict(net_pnl=2.,adverse_selection_cost=1.,mean_squared_inventory=4.)}
    assert policy_cost(result,config,100)==pytest.approx(-.011)


def test_future_does_not_change_prior_quotes(config):
    frames=generate(config,11,'normal')
    changed=copy.deepcopy(frames)
    for f in changed[100:]:
        f['market_buy_qty']=50
    p=baseline('avellaneda-stoikov',config)
    a,b=simulate(frames,config,p),simulate(changed,config,p)
    for x,y in zip(a['trajectory'][:100],b['trajectory'][:100]):
        for key in ('equity','inventory','sigma_ticks','policy_version'):
            assert x[key]==y[key]


@pytest.mark.parametrize('section,key,value',[
    ('data','train_fraction',.9),('data','steps',0),('execution','quote_size',50),
    ('execution','maker_fee_bps',float('nan')),('optimizer','inventory_bands',[100]),
    ('optimizer','spreads',[]),('optimizer','max_evaluations',2),('optimizer','skews',[-1]),
])
def test_config_boundaries(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ValueError):
        validate(config)


def test_cyclic_config(tmp_path):
    p=tmp_path/'bad.yaml'
    p.write_text('extends: bad.yaml\n')
    with pytest.raises(ValueError,match='cyclic'):
        load_config(p)
