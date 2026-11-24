from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rebalancing.engine import plan
from rebalancing.model import InfeasibleError, Portfolio, ResourceLimitError
from rebalancing.solvers import Budget, SolverFailure
from rebalancing.storage import RunStore


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio: dict
    solver: Literal["classical", "greedy", "convex-round", "anneal", "quantum-sim"] = "classical"
    budget: dict = Field(default_factory=dict)
    fallback: Literal["classical"] | None = None


def create_app(directory: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Quantum Rebalancing Research API", version="0.1.0",
                  description="Simulated recommendations only. No exchange or broker connections.")
    # No filesystem side effects at import; initialize storage only on use.
    run_directory = Path(directory or os.environ.get("REBALANCING_RUN_DIR", "runs/api"))

    @app.get("/health")
    def health():
        return {"status": "ok", "execution": "simulated_only"}

    @app.post("/rebalance")
    def rebalance(request: PlanRequest):
        try:
            portfolio = Portfolio.from_dict(request.portfolio)
            budget = Budget(**request.budget)
            result = plan(portfolio, request.solver, budget, fallback=request.fallback)
            run_id = RunStore(run_directory).save(result)
            return {"run_id": run_id, **result}
        except InfeasibleError as exc:
            raise HTTPException(status_code=422, detail={"kind": "infeasible", "message": str(exc)}) from exc
        except ResourceLimitError as exc:
            raise HTTPException(status_code=422, detail={"kind": "resource_limit", "message": str(exc)}) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail={"kind": "invalid_input", "message": str(exc)}) from exc
        except (SolverFailure, ImportError) as exc:
            raise HTTPException(status_code=503, detail={"kind": "solver_failure", "message": str(exc)}) from exc

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        result = RunStore(run_directory).get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return result

    @app.get("/audit/verify")
    def verify_audit():
        return RunStore(run_directory).verify()

    return app


app = create_app()

