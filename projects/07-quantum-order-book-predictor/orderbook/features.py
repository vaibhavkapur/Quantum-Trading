from __future__ import annotations

import numpy as np
from .data import Book

FEATURE_NAMES = ["imbalance_touch", "imbalance_depth", "microprice_offset_bps", "spread_bps",
                 "log_bid_depth", "log_ask_depth", "weighted_imbalance", "ofi_touch",
                 "ofi_5", "return_1_bps", "return_5_bps", "volatility_5_bps"]
WARMUP = 5


def features(book: Book) -> np.ndarray:
    """Every row uses only that snapshot and its five predecessors."""
    bid, ask = book.bid_price[:, 0], book.ask_price[:, 0]
    b, a = book.bid_size[:, 0], book.ask_size[:, 0]
    mid = book.mid
    bd, ad = book.bid_size.sum(1), book.ask_size.sum(1)
    weight = 1 / np.arange(1, book.bid_size.shape[1] + 1)
    wb, wa = book.bid_size @ weight, book.ask_size @ weight
    micro = (ask * b + bid * a) / (b + a)
    ofi = np.zeros(len(book))
    ofi[1:] = ((bid[1:] >= bid[:-1]) * b[1:] - (bid[1:] <= bid[:-1]) * b[:-1]
               - (ask[1:] <= ask[:-1]) * a[1:] + (ask[1:] >= ask[:-1]) * a[:-1])
    # Normalize by depth available before the quote update.
    ofi[1:] /= b[:-1] + a[:-1]
    ret1 = np.zeros(len(book))
    ret1[1:] = (mid[1:] / mid[:-1] - 1) * 1e4
    ret5, ofi5, vol5 = [np.zeros(len(book)) for _ in range(3)]
    for i in range(len(book)):
        ret5[i] = (mid[i] / mid[max(0, i - 5)] - 1) * 1e4
        ofi5[i] = ofi[max(1, i - 4):i + 1].sum()
        vol5[i] = ret1[max(1, i - 4):i + 1].std() if i else 0
    return np.column_stack([(b-a)/(b+a), (bd-ad)/(bd+ad), (micro-mid)/mid*1e4,
                            (ask-bid)/mid*1e4, np.log1p(bd), np.log1p(ad),
                            (wb-wa)/(wb+wa), ofi, ofi5, ret1, ret5, vol5])


def labels(book: Book, horizon: int, threshold_bps: float) -> tuple[np.ndarray, np.ndarray]:
    if type(horizon) is not int or horizon < 1 or not np.isfinite(threshold_bps) or threshold_bps < 0:
        raise ValueError("invalid label horizon or threshold")
    y = np.full(len(book), -1, dtype=int)  # unavailable future is never relabeled as flat
    ends = np.arange(len(book)) + horizon
    if horizon < len(book):
        change = (book.mid[horizon:] / book.mid[:-horizon] - 1) * 1e4
        y[:-horizon] = np.where(change > threshold_bps, 2, np.where(change < -threshold_bps, 0, 1))
    return y, ends
