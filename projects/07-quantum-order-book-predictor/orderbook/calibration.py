import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1):
        raise ValueError("expected finite normalized (n, 3) probabilities")
    return p


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probability, labels):
        p = validate_probabilities(probability)
        logits = np.log(np.clip(p, 1e-12, 1))
        if len(labels) != len(p) or len(labels) == 0 or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("invalid calibration labels")
        def objective(log_temperature):
            calibrated = softmax(logits / np.exp(log_temperature), axis=1)
            return -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        result = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
        self.temperature = float(np.exp(result.x))
        self.objective_evaluations = int(result.nfev)
        return self

    def transform(self, probability):
        p = validate_probabilities(probability)
        return softmax(np.log(np.clip(p, 1e-12, 1)) / self.temperature, axis=1)
