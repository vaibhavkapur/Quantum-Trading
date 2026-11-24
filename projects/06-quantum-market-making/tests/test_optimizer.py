from concurrent.futures import Future
from dataclasses import FrozenInstanceError
import threading
import time
import numpy as np
import pytest
from market_making.data import generate
from market_making.optimizer import encode_costs, qaoa_probabilities, quantum_search, optimize, PolicyService
from market_making.policy import Policy, baseline


def test_policy_immutable():
    policy=Policy(1,0,4)
    with pytest.raises(FrozenInstanceError):
        policy.version=1


def test_encoding_padding_and_known_optimum():
    energies=encode_costs([3,-2,1])
    np.testing.assert_allclose(energies,[1,0,.6,2])
    assert np.argmin(energies)==1


@pytest.mark.parametrize('costs',[[],[float('nan')],[float('inf')],[[1,2]]])
def test_encoding_bad_input(costs):
    with pytest.raises(ValueError):
        encode_costs(costs)


def test_qaoa_against_direct_matrix_evolution():
    energies=encode_costs([2,0,1])
    gamma,beta=.7,.2
    state=np.ones(4,dtype=complex)/2*np.exp(-1j*gamma*energies)
    rx=np.array([[np.cos(beta),-1j*np.sin(beta)],[-1j*np.sin(beta),np.cos(beta)]])
    expected=np.abs(np.kron(rx,rx)@state)**2
    np.testing.assert_allclose(qaoa_probabilities(energies,[gamma,beta]),expected,atol=1e-12)
    np.testing.assert_allclose(qaoa_probabilities(energies,[0,0]),[.25]*4,atol=1e-12)


def test_quantum_known_optimum_shots_and_reproducibility(config):
    args=([2.,0.,1.],11,config['optimizer'])
    a,details=quantum_search(*args,time.perf_counter()+15)
    b,other=quantum_search(*args,time.perf_counter()+15)
    assert a==b==1
    assert details['shots']==64
    assert details['expectation_evaluations']<=12
    assert details['optimal_probability']==pytest.approx(other['optimal_probability'])
    assert not details['hardware']


def test_classical_quantum_identical_objective_and_constraints(config):
    frames=generate(config,11,'normal')[:30]
    classic,cd=optimize(frames,config,'classical',11,1)
    quantum,qd=optimize(frames,config,'quantum-sim',11,1)
    assert cd['costs']==qd['costs']
    assert cd['selected_index']==int(np.argmin(cd['costs']))
    assert cd['optimality_gap']==0
    assert qd['selected_cost']>=cd['selected_cost']-1e-12
    assert qd['objective_evaluations']==cd['objective_evaluations']==8
    assert quantum.inventory_band<=config['execution']['max_inventory']


def test_failure_fallback_and_monotonic_version(config):
    def fail(*args):
        raise RuntimeError('injected failure')
    frames=generate(config,11,'normal')[:10]
    with PolicyService(config,'classical',11,worker=fail) as service:
        p=service.update(frames,baseline('inventory-skew',config),frames[-1]['timestamp_ns']+1)
        assert p.source=='fallback' and p.version==1
        p=service.update(frames,p,frames[-1]['timestamp_ns']+2)
        assert p.version==2
        assert len(service.records)==2
        assert 'injected failure' in service.records[0]['reason']


def test_nonblocking_poll_timeout_and_discard_late_result(config):
    gate=threading.Event()
    config['optimizer']['timeout_seconds']=.03
    def slow(frames,config,solver,seed,version,deadline):
        gate.wait(1)
        return Policy(1,0,4,version),{'status':'ok'}
    frames=generate(config,11,'normal')[:10]
    with PolicyService(config,'classical',11,worker=slow) as service:
        service.request(frames,10**10)
        assert service.poll(10**10) is None
        fallback=service.poll(10**10,wait=True)
        assert fallback.source=='fallback'
        gate.set()
        service.pending[0].result(timeout=1)
        assert service.poll(10**10+1) is None
        assert service.version==fallback.version


def test_reject_invalid_policy_and_temporal_leakage(config):
    frames=generate(config,11,'normal')[:10]
    def invalid(*args):
        return Policy(1,0,100,1),{'status':'ok'}
    with PolicyService(config,'classical',11,worker=invalid) as service:
        with pytest.raises(ValueError,match='strictly precede'):
            service.request(frames,frames[-1]['timestamp_ns'])
        policy=service.update(frames,baseline('inventory-skew',config),10**10)
        assert policy.source=='fallback'


def test_qiskit_across_recreated_workers(config):
    frames=generate(config,11,'normal')[:10]
    for _ in range(2):
        with PolicyService(config,'quantum-sim',11) as service:
            p=service.update(frames,baseline('inventory-skew',config),10**10)
            assert p.source=='quantum-sim'
