from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from rebalancing.data import generate
from rebalancing.encoding import encode, make_qubo
from rebalancing.engine import plan
from rebalancing.model import InfeasibleError, Problem, ResourceLimitError
from rebalancing.solvers import Budget, Candidate, anneal, brute_force, convex_round, greedy, repair, solve_milp


def test_qubo_and_ising_exhaustively(tiny):
    problem = Problem(tiny)
    e = encode(problem)
    q = make_qubo(e)
    constant, fields, interactions = q.ising()
    minimum, minimum_valid = (float('inf'),None), float('inf')
    for raw in product((0,1),repeat=e.size):
        x = np.array(raw)
        z = 1-2*x
        penalty = q.penalty * sum((x[b].sum()-1)**2 for b in e.blocks)
        assert q.energy(x) == pytest.approx(e.objective(x) + penalty, abs=1e-12)
        assert q.energy(x) == pytest.approx(constant + fields @ z + z @ interactions @ z,abs=1e-12)
        valid = all(x[b].sum()==1 for b in e.blocks)
        if q.energy(x)<minimum[0]: minimum=(q.energy(x),valid)
        if valid:
            assert e.objective(x) == pytest.approx(problem.evaluate(e.decode(x))['objective'],abs=1e-12)
            minimum_valid=min(minimum_valid,q.energy(x))
    assert minimum[1]
    assert minimum[0] == pytest.approx(minimum_valid)


def test_known_milp_optimum(known):
    p=Problem(known)
    solved=solve_milp(encode(p),Budget())
    exact=brute_force(p)
    assert solved.status == 'optimal'
    assert solved.indices == exact.indices
    assert p.evaluate(solved.indices)['objective']==pytest.approx(0)
    assert solved.metadata['objective_lower_bound']==pytest.approx(0,abs=1e-10)


@pytest.mark.parametrize('seed',range(8))
def test_milp_matches_exhaustive_under_identical_constraints(seed):
    p=Problem(generate(seed,3,1))
    exact=brute_force(p)
    actual=solve_milp(encode(p),Budget())
    assert p.evaluate(actual.indices)['objective']==pytest.approx(p.evaluate(exact.indices)['objective'],abs=1e-10)


@pytest.mark.parametrize('solver',['greedy','convex-round','anneal','classical','quantum-sim'])
def test_all_solver_pipelines_are_feasible_and_conserve_cash(tiny,solver):
    result=plan(tiny,solver,Budget(max_evaluations=12,shots=96))
    r=result['report']
    assert r['feasible'] and r['conservation_residual_cents']==0
    assert r['cash_after_cents']>=tiny.reserve_cents
    assert r['post_nav_cents']+r['fees_cents']+r['taxes_cents']==r['initial_nav_cents']
    assert result['fallback_reason'] is None


def test_repair_minimizes_lot_distance_and_is_not_tracking_optimizer(tiny):
    p=Problem(replace(tiny,cash_reserve_cents=16500))
    e=encode(p)
    raw=Candidate((2,2),'test')
    fixed, info=repair(e,raw,Budget())
    assert info['applied'] and p.evaluate(fixed)['feasible']
    desired=[opts[j].delta_lots for opts,j in zip(p.choices,raw.indices)]
    distances=[]
    for indices in product(*(range(len(c)) for c in p.choices)):
        if p.evaluate(indices)['feasible']:
            distances.append(sum(abs(opts[j].delta_lots-d) for opts,j,d in zip(p.choices,indices,desired)))
    assert info['distance_lots']==min(distances)


def test_infeasible_has_no_silent_orders(tiny):
    p=replace(tiny,cash_reserve_cents=23000)
    with pytest.raises(InfeasibleError):
        plan(p,'classical')
    with pytest.raises(InfeasibleError):
        plan(p,'greedy')


def test_limits_and_explicit_fallback(tiny):
    budget=Budget(max_qubits=1)
    with pytest.raises(ResourceLimitError):
        plan(tiny,'quantum-sim',budget)
    result=plan(tiny,'quantum-sim',budget,fallback='classical')
    assert result['solver_requested']=='quantum-sim' and result['solver_used']=='classical'
    assert 'ResourceLimitError' in result['fallback_reason']
    assert result['report']['feasible']
    with pytest.raises(ResourceLimitError):
        brute_force(Problem(tiny),max_combinations=1)


def test_missing_quantum_dependency_has_explicit_fallback(tiny,monkeypatch):
    import rebalancing.engine as engine
    def missing(*args): raise ImportError('test missing qiskit')
    monkeypatch.setitem(engine.SOLVERS,'quantum-sim',missing)
    with pytest.raises(ImportError): plan(tiny,'quantum-sim')
    assert plan(tiny,'quantum-sim',fallback='classical')['solver_used']=='classical'


def test_seeded_qaoa_is_reproducible_and_honors_evaluation_budget(tiny):
    budget=Budget(seed=41,max_evaluations=9,shots=64)
    a,b=plan(tiny,'quantum-sim',budget),plan(tiny,'quantum-sim',budget)
    assert a['recommendation_id']==b['recommendation_id']
    assert a['solver_metadata']['top_counts']==b['solver_metadata']['top_counts']
    assert a['evaluations']<=9
    assert a['solver_metadata']['optimizer_shots']==0
    assert a['solver_metadata']['hardware'] is False
    assert a['solver_metadata']['qubits']==6


@pytest.mark.parametrize('solver',[anneal,convex_round])
def test_iterative_evaluation_limit(tiny,solver):
    result=solver(encode(Problem(tiny)),Budget(max_evaluations=1))
    assert result.evaluations==1


@pytest.mark.parametrize('changes',[{'max_evaluations':0},{'shots':0},{'seed':-1},{'time_limit_s':0},
    {'max_qubits':21},{'repair_time_limit_s':float('nan')},{'qaoa_depth':0}])
def test_invalid_budgets(changes):
    with pytest.raises(ValueError): Budget(**changes)

