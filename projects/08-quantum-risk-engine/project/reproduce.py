"""One CPU-only command to regenerate data, both runs, benchmark and report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import fingerprint, load_config
from .data import prepare_dataset
from .report import build_report
from .run import execute
from .workflow import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="reports/reproduction")
    args = parser.parse_args()
    config, root = load_config(args.config)
    destination = Path(root) / args.output
    destination.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(config, root)
    run_paths = [execute(config, root, solver, str(destination / "runs")) for solver in ("classical", "quantum-sim")]
    from benchmarks.run import run_benchmark
    benchmark = run_benchmark(config, root, str(destination / "benchmark"))
    report = build_report(destination / "runs", destination / "comparison.html", destination / "benchmark.json")
    sources = {}
    for directory in ["risk", "scenarios", "quantum", "project", "api", "benchmarks", "native", "tests"]:
        for path in sorted((Path(__file__).parents[1] / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".html", ".txt"):
                sources[str(path.relative_to(Path(__file__).parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = [prepared, *run_paths, report, report.with_suffix(".csv"), destination / "benchmark.json", destination / "benchmark.csv", destination / "scaling.svg"]
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "command": "python -m project.reproduce --config " + args.config + " --output " + args.output,
                "config": config.model_dump(), "config_sha256": fingerprint(config.model_dump()),
                "environment": environment(), "source_sha256": sources,
                "artifacts": [{"path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
                               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts],
                "benchmark_failed_trials": sum(r["status"] != "ok" for r in benchmark["trials"])}
    (destination / "experiment-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(report)
    if manifest["benchmark_failed_trials"]:
        raise SystemExit("Benchmark has failed trials; inspect the saved manifest and raw results")


if __name__ == "__main__":
    main()
