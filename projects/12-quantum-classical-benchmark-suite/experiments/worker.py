"""One isolated process per attempt, including discarded warmup attempts."""
from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path


def execute(request: dict) -> dict:
    started = time.perf_counter()
    from benchmarks.adapters import get_adapter
    from benchmarks.adapters.base import Budget, BudgetExceeded, Context
    from benchmarks.instances.dataset import load_instance
    from benchmarks.instances.schema import feasible, objective, repair

    ctx = Context(Budget(**request["budget"]), request["seed"], request["algorithm"])
    ctx.start = started
    ctx.timings["setup"] = time.perf_counter() - started
    outcome, status, error, instance = {}, "ok", None, None
    try:
        with ctx.stage("data_prep"):
            instance = load_instance(Path(request["instance_path"]), request["instance_sha256"])
        with ctx.stage("setup"):
            adapter = get_adapter(request["solver"])
        ctx.check()
        # Only the common problem specification is passed to the adapter.
        outcome = adapter.solve(instance["problem"], ctx)
        status = outcome.pop("solver_status", "ok")
        ctx.check()
    except BudgetExceeded as exc:
        outcome, status, error = ctx.partial, "budget_exhausted", str(exc)
    except Exception as exc:
        outcome, status, error = {}, "failed", f"{type(exc).__name__}: {exc}"
    quality = {"feasible": False, "raw_feasible": False, "repair_applied": False,
               "objective": None, "absolute_gap": None, "relative_gap": None,
               "estimate": None, "absolute_error": None, "success": False, "interval_covers_reference": None}
    try:
        if instance is not None:
            p, reference = instance["problem"], instance["reference"]["value"]
            quality["reference"] = reference
            if p["track"] != "estimation":
                raw = outcome.get("candidate")
                with ctx.stage("validation"):
                    quality["raw_feasible"] = feasible(p, raw)
                if raw is not None and not quality["raw_feasible"] and request["algorithm"]["repair"]:
                    with ctx.stage("repair"):
                        outcome["raw_candidate"] = raw
                        outcome["candidate"] = repair(p, raw)
                        quality["repair_applied"] = True
                with ctx.stage("validation"):
                    quality["feasible"] = feasible(p, outcome.get("candidate"))
                    if quality["feasible"]:
                        value = objective(p, outcome["candidate"])
                        gap = value - reference
                        if gap < -1e-8:
                            raise ValueError("solver beat the verified reference; dataset or evaluator is inconsistent")
                        quality.update(objective=value, absolute_gap=max(0., gap),
                                       relative_gap=max(0., gap) / max(abs(reference), 1e-12), success=abs(gap) <= 1e-8)
            else:
                with ctx.stage("validation"):
                    estimate = outcome.get("estimate")
                    if estimate is not None and math.isfinite(estimate) and 0 <= estimate <= p["scale"]:
                        error_value = abs(estimate - reference)
                        quality.update(feasible=True, raw_feasible=True, estimate=estimate, absolute_error=error_value,
                                       success=error_value <= request["success_tolerance"])
                        interval = outcome.get("estimate_ci95")
                        if interval is not None:
                            quality["interval_covers_reference"] = interval[0] <= reference <= interval[1]
            if status == "ok" and not quality["feasible"]:
                status = "infeasible" if p["track"] != "estimation" else "failed"
    except Exception as exc:
        status, error = "failed", f"validation: {type(exc).__name__}: {exc}"
        quality.update(feasible=False, success=False, objective=None, estimate=None)
    if status == "failed":
        quality.update(success=False, feasible=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    worker_seconds = time.perf_counter() - started
    return {"status": status, "error": error, "outcome": outcome, "quality": quality,
            "resources": ctx.resources | {"peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                                           "memory_method": "isolated process ru_maxrss; includes interpreter/native libraries"},
            "timings": ctx.timings | {"worker_seconds": worker_seconds,
                                      "worker_overhead": max(0., worker_seconds - sum(ctx.timings.values()))}}


def main() -> None:
    request_path, response_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    result = execute(request)
    temporary = response_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.replace(response_path)


if __name__ == "__main__":
    main()
