"""Out-of-sample portfolio metrics and small statistics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance(daily: pd.DataFrame, ann: float = 252.0) -> dict:
    """Metrics from a frame with columns ``ret`` (net of costs) and ``bench``."""
    r = daily["ret"].to_numpy()
    if len(r) < 2:
        return {"n_days": int(len(r))}
    nav = np.cumprod(1 + r)
    vol = r.std(ddof=1) * np.sqrt(ann)
    active = r - daily["bench"].to_numpy()
    return {
        "n_days": int(len(r)),
        "ann_return": float(nav[-1] ** (ann / len(r)) - 1),
        "ann_vol": float(vol),
        "sharpe": float(r.mean() * ann / vol) if vol > 0 else float("nan"),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "tracking_error": float(active.std(ddof=1) * np.sqrt(ann)),
        "bench_ann_return": float(np.prod(1 + daily["bench"].to_numpy()) ** (ann / len(r)) - 1),
    }


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and a Student-t confidence interval across seeds. ``(mean, lo, hi)``."""
    from scipy import stats

    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    half = float(stats.t.ppf(0.5 + level / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - half, m + half
