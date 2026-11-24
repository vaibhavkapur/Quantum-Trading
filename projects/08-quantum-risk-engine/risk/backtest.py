import numpy as np
from scipy.special import xlogy
from scipy.stats import beta, binomtest, chi2

from .measures import empirical_risk, gaussian_risk, validate_alpha
from .models import finite_vector


def _binomial_log_likelihood(successes, total, probability):
    return float(xlogy(successes, probability) + xlogy(total - successes, 1 - probability))


def coverage_statistics(losses, forecasts, alpha):
    validate_alpha(alpha)
    losses, forecasts = finite_vector(losses), finite_vector(forecasts, "forecasts")
    if forecasts.shape != losses.shape:
        raise ValueError("one forecast is required per realized loss")
    flags = losses > forecasts  # strict breach, not equality
    n, k = len(flags), int(flags.sum())
    lr = max(0.0, 2 * (_binomial_log_likelihood(k, n, k/n) - _binomial_log_likelihood(k, n, 1-alpha)))
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k+1, n-k))
    transitions = np.zeros((2, 2), dtype=int)
    for previous, current in zip(flags[:-1], flags[1:]):
        transitions[int(previous), int(current)] += 1
    independence_p = None
    if np.all(transitions.sum(axis=1) > 0):
        total = n - 1
        ones = int(transitions[:, 1].sum())
        null_ll = _binomial_log_likelihood(ones, total, ones / total)
        alt_ll = sum(_binomial_log_likelihood(int(row[1]), int(row.sum()), row[1] / row.sum()) for row in transitions)
        independence_p = float(chi2.sf(max(0, 2 * (alt_ll - null_ll)), 1))
    return {"observations": n, "exceptions": k, "exception_rate": k/n,
            "expected_exception_rate": 1-alpha, "exception_rate_ci95": [lo, hi],
            "kupiec_lr": lr, "kupiec_pvalue": float(chi2.sf(lr, 1)),
            "exact_binomial_pvalue": float(binomtest(k, n, 1-alpha).pvalue),
            "independence_pvalue": independence_p, "transitions": transitions.tolist()}


def rolling_backtest(losses, window=250, alpha=0.99, method="historical"):
    losses = finite_vector(losses, "losses")
    validate_alpha(alpha)
    if not isinstance(window, int) or isinstance(window, bool) or not 2 <= window < len(losses):
        raise ValueError("window must be an integer from 2 to len(losses)-1")
    if method not in ("historical", "gaussian"):
        raise ValueError("method must be historical or gaussian")
    forecasts = []
    for i in range(window, len(losses)):
        past = losses[i-window:i]  # excludes the observation being forecast
        result = (empirical_risk(past, alpha) if method == "historical" else
                  gaussian_risk(float(past.mean()), float(past.std(ddof=1)), alpha))
        forecasts.append(result["var"])
    realized = losses[window:]
    return {"method": method, "window": window, "alpha": alpha,
            "coverage": coverage_statistics(realized, forecasts, alpha),
            "forecasts": forecasts, "losses": realized.tolist(),
            "exception_indices": (np.flatnonzero(realized > forecasts) + window).tolist()}
