from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
import pickle
import numpy as np

from .data import Book
from .features import features, WARMUP
from .evaluation import decision


@dataclass
class Predictor:
    reducer: object
    model: object
    calibrator: object
    threshold: float
    levels: int

    def predict_features(self, x):
        return self.calibrator.transform(self.model.predict_proba(self.reducer.transform(x)))

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self, protocol=5))

    @staticmethod
    def load(path):
        # Pickle is only appropriate for trusted, locally produced model artifacts.
        result = pickle.loads(Path(path).read_bytes())
        if not isinstance(result, Predictor):
            raise ValueError("artifact is not a Predictor")
        return result


def snapshot(book, index):
    return {"timestamp_ms": int(book.timestamp_ms[index]),
            **{name: getattr(book, name)[index].tolist()
               for name in ("bid_price", "ask_price", "bid_size", "ask_size")}}


class StreamingPredictor:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self.history = deque(maxlen=WARMUP + 1)

    def update(self, row):
        rows = [*self.history, row][-(WARMUP+1):]
        timestamp = np.array([r["timestamp_ms"] for r in rows])
        book = Book(timestamp, *[np.asarray([r[name] for r in rows], dtype=float)
                               for name in ("bid_price", "ask_price", "bid_size", "ask_size")]).validate()
        if book.bid_price.shape[1] != self.predictor.levels:
            raise ValueError("stream depth does not match trained model")
        self.history.append(snapshot(book, -1))
        if len(rows) <= WARMUP:
            return {"timestamp_ms": int(timestamp[-1]), "status": "warming_up", "direction": 0}
        p = self.predictor.predict_features(features(book)[-1:])[0]
        return {"timestamp_ms": int(timestamp[-1]), "status": "ok", "probabilities": p.tolist(),
                "direction": decision(p, self.predictor.threshold)}

    def update_safe(self, row):
        try:
            return self.update(row)
        except Exception as error:
            return {"status": "error", "direction": 0, "error": f"{type(error).__name__}: {error}"}


def profile_stream(predictor, book, indices, count):
    stream = StreamingPredictor(predictor)
    start = int(indices[0])
    for i in range(start - WARMUP, start):
        stream.update(snapshot(book, i))
    timings = []
    for i in indices[:count]:
        begin = perf_counter_ns()
        row = snapshot(book, int(i))
        stream.update(row)
        timings.append((perf_counter_ns() - begin) / 1e6)
    a = np.asarray(timings)
    return {"samples": len(a), "p50_ms": float(np.quantile(a, 0.5)),
            "p95_ms": float(np.quantile(a, 0.95)), "p99_ms": float(np.quantile(a, 0.99)),
            "max_ms": float(a.max()), "throughput_per_second": float(1000 / a.mean()),
            "scope": "snapshot conversion + validation + causal features + scaling/PCA + model + calibration + decision",
            "includes_network": False}, timings
