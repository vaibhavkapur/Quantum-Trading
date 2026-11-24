import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .config import fingerprint, load_config
from .data import load_dataset
from .workflow import calculate, environment


def execute(config, root, solver, output="runs"):
    started = perf_counter()
    returns, manifest = load_dataset(config, root)
    loaded = perf_counter()
    result = calculate(config, returns, solver)
    run = {"schema_version": 1, "run_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
           "experiment": config.experiment, "seed": config.seed, "config": config.model_dump(),
           "config_sha256": fingerprint(config.model_dump()), "data_sha256": manifest["sha256"],
           "environment": environment(), **result}
    run["timings"]["load_seconds"] = loaded-started
    run["timings"]["end_to_end_seconds"] = perf_counter()-started
    run["timings"]["timing_boundary"] = "Data load through result/metadata assembly; JSON disk write excluded"
    directory = Path(root) / output
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.experiment}-{solver}-{run['run_id']}.json"
    with path.open("x") as handle:
        json.dump(run, handle, indent=2, allow_nan=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run classical risk or the Qiskit simulator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=["classical", "quantum-sim"], required=True)
    parser.add_argument("--output", default="runs")
    args = parser.parse_args()
    config, root = load_config(args.config)
    print(execute(config, root, args.solver, args.output))


if __name__ == "__main__":
    main()
