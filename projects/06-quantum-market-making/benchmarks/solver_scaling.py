"""Compare search-only scaling on identical, explicitly enumerated objectives.

The application benchmark includes expensive replay-based objective preparation.
This separate microbenchmark measures circuit search at 2..6 qubits.
"""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from market_making.config import load_config
from market_making.experiment import write_csv
from market_making.optimizer import quantum_search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/solver_scaling.csv')
    args=parser.parse_args()
    config=load_config(Path(__file__).parents[1]/'configs/demo.yaml')
    rows=[]
    for n in (4,8,16,32,64):
        for seed in config['seeds']:
            begin=time.perf_counter()
            # Same known quadratic objective for each approach; optimum is n//3.
            costs=(np.arange(n,dtype=float)-n//3)**2/max(n*n,1)
            prep=time.perf_counter()-begin
            begin=time.perf_counter()
            index=int(np.argmin(costs))
            rows.append(dict(policies=n,seed=seed,solver='classical',gap=float(costs[index]-min(costs)),
                             input_preparation_seconds=prep,search_seconds=time.perf_counter()-begin,
                             qubits=0,shots=0,expectation_evaluations=0))
            begin=time.perf_counter()
            index,detail=quantum_search(costs,seed,config['optimizer'],time.perf_counter()+30)
            rows.append(dict(policies=n,seed=seed,solver='quantum-sim',gap=float(costs[index]-min(costs)),
                             input_preparation_seconds=prep,search_seconds=time.perf_counter()-begin,
                             qubits=detail['qubits'],shots=detail['shots'],expectation_evaluations=detail['expectation_evaluations']))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    write_csv(args.output,rows)
    print(args.output)


if __name__=='__main__':
    main()
