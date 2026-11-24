from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from .calibration import validate_probabilities
from .config import Config
from .data import Book


def classification_metrics(y, probabilities):
    p = validate_probabilities(probabilities)
    predicted = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1, 2], zero_division=0)
    confidence = p.max(1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean())
    return {
        "balanced_accuracy": float(recall[support > 0].mean()), "macro_f1": float(f1.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_score": float(np.square(p - np.eye(3)[y]).sum(1).mean()), "ece": float(ece),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(["down", "flat", "up"])},
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
    }


def decision(probability, threshold):
    label = int(np.argmax(probability))
    if threshold >= 1 or probability[label] < threshold or label == 1:
        return 0
    return -1 if label == 0 else 1


def backtest(book: Book, indices, probabilities, threshold: float, config: Config):
    """One base unit, taker execution, delayed entry, non-overlapping fixed-horizon exits."""
    p = validate_probabilities(probabilities)
    indices = np.asarray(indices, dtype=int)
    if len(indices) != len(p) or np.any(np.diff(indices) <= 0):
        raise ValueError("signals must align with unique chronological indices")
    if len(indices) and (indices.min() < 0 or indices.max() + config.horizon >= len(book)):
        raise ValueError("signal horizon exceeds available book")
    previous_exit, trades = -1, []
    for i, probability in zip(indices, p):
        direction = decision(probability, threshold)
        entry, exit_ = int(i + config.execution_delay), int(i + config.horizon)
        if direction == 0 or entry <= previous_exit:
            continue
        # Refuse to assume more top-of-book liquidity than is present.
        entry_size = book.ask_size[entry, 0] if direction == 1 else book.bid_size[entry, 0]
        exit_size = book.bid_size[exit_, 0] if direction == 1 else book.ask_size[exit_, 0]
        if min(entry_size, exit_size) < 1:
            raise ValueError("one-unit execution exceeds displayed depth; reduce unit size/data scope")
        mid_entry, mid_exit = book.mid[entry], book.mid[exit_]
        entry_price = book.ask_price[entry, 0] if direction == 1 else book.bid_price[entry, 0]
        exit_price = book.bid_price[exit_, 0] if direction == 1 else book.ask_price[exit_, 0]
        entry_price *= 1 + direction * config.slippage_bps / 1e4
        exit_price *= 1 - direction * config.slippage_bps / 1e4
        gross = direction * (mid_exit - mid_entry) / mid_entry * 1e4
        execution = direction * (exit_price - entry_price) / mid_entry * 1e4
        fees = config.fee_bps * (entry_price + exit_price) / mid_entry
        net = execution - fees
        trades.append({"signal_index": int(i), "entry_index": entry, "exit_index": exit_,
                       "entry_timestamp_ms": int(book.timestamp_ms[entry]),
                       "exit_timestamp_ms": int(book.timestamp_ms[exit_]), "direction": direction,
                       "entry_price": float(entry_price), "exit_price": float(exit_price),
                       "gross_bps": float(gross), "spread_slippage_bps": float(gross-execution),
                       "fees_bps": float(fees), "net_bps": float(net),
                       "net_quote_currency": float(net * mid_entry / 1e4)})
        previous_exit = exit_
    returns = np.array([t["net_bps"] for t in trades])
    curve = np.r_[0, returns.cumsum()]
    return {
        "trades": len(trades), "gross_bps": float(sum(t["gross_bps"] for t in trades)),
        "net_bps": float(returns.sum()), "mean_trade_bps": float(returns.mean()) if len(returns) else 0.0,
        "net_quote_currency": float(sum(t["net_quote_currency"] for t in trades)),
        "max_drawdown_bps": float(np.max(np.maximum.accumulate(curve) - curve)),
        "hit_rate": float(np.mean(returns > 0)) if len(returns) else 0.0,
        "turnover_units": 2 * len(trades), "threshold": threshold,
    }, trades


def select_threshold(book, indices, probabilities, config):
    trials = []
    for threshold in sorted(set([*config.thresholds, 1.0])):
        result, _ = backtest(book, indices, probabilities, threshold, config)
        trials.append(result)
    # Ties prefer abstention / the more conservative threshold.
    best = max(trials, key=lambda row: (row["net_bps"], row["threshold"]))
    return best["threshold"], trials
