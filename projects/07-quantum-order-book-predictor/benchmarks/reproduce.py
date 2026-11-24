"""Run from the project root: python benchmarks/reproduce.py --config configs/ci.yaml."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ci.yaml")
    parser.add_argument("--output", default="reports/comparison.html")
    args = parser.parse_args()
    # Invocations use exactly the public CLI and the current interpreter.
    commands = [
        ["project.prepare_data", "--config", args.config],
        ["project.run", "--config", args.config, "--solver", "classical"],
        ["project.run", "--config", args.config, "--solver", "quantum-sim"],
    ]
    created = []
    for command in commands:
        result = subprocess.run([sys.executable, "-m", *command], text=True, capture_output=True)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if command[0] == "project.run" and result.stdout.strip():
            candidate = Path(result.stdout.strip().splitlines()[-1])
            if candidate.name == "manifest.json" and candidate.exists():
                created.append(candidate)
        if result.returncode and not created:
            raise SystemExit(result.returncode)
    # Stage only this invocation's manifests to avoid counting repeated seeds twice.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        for i, path in enumerate(created):
            destination = Path(temp) / str(i)
            destination.mkdir()
            (destination / "manifest.json").symlink_to(path.resolve())
        result = subprocess.run([sys.executable, "-m", "project.report", "--runs", temp, "--output", args.output])
    import json
    failed = any(json.loads(p.read_text())["status"] != "complete" for p in created)
    raise SystemExit(result.returncode or int(failed or len(created) != 2))


if __name__ == "__main__":
    main()
