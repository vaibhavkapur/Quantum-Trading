from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import Field

from project.config import ExperimentConfig, ModelConfig, PortfolioConfig, StrictModel, fingerprint
from project.workflow import calculate
from risk.backtest import rolling_backtest
from risk.measures import empirical_risk, gaussian_risk
from scenarios import correlated_returns

app = FastAPI(title="Quantum Risk Engine", version="0.1.0",
              description="One-day synthetic risk research. Positive values represent losses. Currency follows portfolio value.")


class LossRequest(StrictModel):
    losses: List[float] = Field(min_length=1, max_length=200000)
    alpha: float = Field(default=0.99, gt=0, lt=1)
    weights: Optional[List[float]] = Field(default=None, max_length=200000)


class GaussianRequest(StrictModel):
    mean_loss: float = 0
    std_loss: float = Field(ge=0)
    alpha: float = Field(default=0.99, gt=0, lt=1)


class StressRequest(StrictModel):
    portfolio: PortfolioConfig
    shocks: Dict[str, List[float]]


class BacktestRequest(StrictModel):
    losses: List[float] = Field(min_length=3, max_length=5000)
    window: int = Field(default=250, ge=2, le=2000)
    alpha: float = Field(default=0.99, gt=0, lt=1)
    method: Literal["historical", "gaussian"] = "historical"


class RunRequest(StrictModel):
    config: ExperimentConfig
    solver: Literal["classical", "quantum-sim"] = "classical"


@app.exception_handler(ValueError)
async def invalid_value(request: Request, error: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/health")
def health():
    return {"status": "ok", "quantum_available": find_spec("qiskit") is not None, "hardware": False}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path(__file__).with_name("dashboard.html").read_text()


@app.get("/api/example")
def example():
    return ExperimentConfig(experiment="synthetic-three-asset-v1",
        portfolio=PortfolioConfig(assets=["EQUITY", "BONDS", "GOLD"], weights=[0.6, 0.3, 0.1]),
        model=ModelConfig(mean=[0.0003, 0.0001, 0.00015], covariance=[
            [0.000144, 0.0000108, 0.0000144], [0.0000108, 0.000036, -0.0000048],
            [0.0000144, -0.0000048, 0.000064]]),
        stress={"Equity selloff": [-0.15, 0.02, 0.04], "Rates shock": [-0.06, -0.08, -0.02],
                "Liquidity shock": [-0.12, -0.05, -0.09]}).model_dump()


@app.post("/api/risk/historical")
def historical(request: LossRequest):
    return empirical_risk(request.losses, request.alpha, request.weights)


@app.post("/api/risk/gaussian")
def gaussian(request: GaussianRequest):
    return gaussian_risk(request.mean_loss, request.std_loss, request.alpha)


@app.post("/api/stress")
def stress(request: StressRequest):
    if len(request.shocks) > 100:
        raise HTTPException(status_code=422, detail="at most 100 stress scenarios")
    return request.portfolio.build().stress(request.shocks)


@app.post("/api/backtest")
def backtest(request: BacktestRequest):
    return rolling_backtest(request.losses, request.window, request.alpha, request.method)


@app.post("/api/run")
def run(request: RunRequest):
    config = request.config
    # Keep interactive requests bounded; larger CLI experiments remain available.
    if config.data.observations > 5000 or config.quantum.qubits > 5 or max(config.quantum.powers) > 16:
        raise HTTPException(status_code=422, detail="API limits: 5000 observations, 5 state qubits, Grover power 16")
    model = config.model
    returns = correlated_returns(model.mean, model.covariance, config.data.observations, config.seed,
                                 model.distribution, model.df)
    try:
        result = calculate(config, returns, request.solver)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"config_sha256": fingerprint(config.model_dump()), "config": config.model_dump(), **result}
