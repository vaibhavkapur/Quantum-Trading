"""Checked ctypes boundary to the C++20 matching, quoting and hard-risk engine."""
from __future__ import annotations

import ctypes as C
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time
from functools import lru_cache


class Fill(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ("timestamp_ns", "order_id", "side", "price", "quantity", "maker")] + [("fee", C.c_double)]


class Stats(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ("inventory", "open_buy", "open_sell", "cancels", "fills", "halted", "policy_version")] + [(k, C.c_double) for k in ("cash", "maker_fees", "taker_fees", "peak_equity")]

    def as_dict(self):
        return {k: getattr(self, k) for k, _ in self._fields_}


def build() -> tuple[Path, float]:
    source = Path(__file__).parent / "native" / "engine.cpp"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-std=c++20", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-Wall", "-Wextra", "-Wpedantic"]
    key = hashlib.sha256(source.read_bytes() + repr((compiler, flags, platform.platform())).encode()).hexdigest()[:16]
    directory = Path(os.environ.get("MM_BUILD_DIR", Path(__file__).parent.parent / "build"))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("engine-" + key + (".dylib" if platform.system() == "Darwin" else ".so"))
    start = time.perf_counter()
    with (directory / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            subprocess.run(compiler + flags + [str(source), "-o", str(temporary)], check=True, capture_output=True, text=True)
            temporary.replace(target)
    return target, time.perf_counter() - start


@lru_cache(maxsize=1)
def library():
    path, _ = build()
    lib = C.CDLL(str(path))
    p, i, d = C.c_void_p, C.c_int64, C.c_double
    signatures = {
        "create": ([d, d, d, d, i], p), "destroy": ([p], None), "error": ([], C.c_char_p),
        "add": ([p, i, i, i, i, i, C.c_int, C.c_int], i), "cancel": ([p, i], i),
        "cancel_all": ([p, C.c_int], i), "queue": ([p, i], i),
        "depth": ([p, C.c_int, i, i], i), "stats": ([p, C.POINTER(Stats)], i),
        "fill": ([p, i, C.POINTER(Fill)], i), "flatten": ([p, i], i),
        "quote": ([p, i, d, d, d, i, i, d, d, d, d, i], i),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, "mm_" + name)
        fn.argtypes, fn.restype = args, result
    return lib


class Book:
    def __init__(self, execution):
        self.lib = library()
        self.execution = execution
        self.ptr = self.lib.mm_create(execution["tick_value"], execution["maker_fee_bps"], execution["taker_fee_bps"], execution["cancel_cost"], execution["max_inventory"])
        if not self.ptr:
            raise ValueError(self.lib.mm_error().decode())
        self.depth_prices = {1: set(), -1: set()}

    def close(self):
        if self.ptr:
            self.lib.mm_destroy(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def call(self, name, *args):
        if not self.ptr:
            raise RuntimeError("book is closed")
        value = getattr(self.lib, "mm_" + name)(self.ptr, *args)
        if value == -1:
            raise ValueError(self.lib.mm_error().decode())
        return value

    def add(self, order_id, side, price, quantity, own=False, timestamp_ns=0, market=False):
        return self.call("add", timestamp_ns, order_id, side, price, quantity, own, market)

    def cancel(self, order_id):
        return bool(self.call("cancel", order_id))

    def cancel_all(self, emergency=False):
        self.call("cancel_all", emergency)

    def queue_ahead(self, order_id):
        if not self.ptr:
            raise RuntimeError("book is closed")
        return self.lib.mm_queue(self.ptr, order_id)

    def reconcile(self, frame):
        for side, name in ((1, "bids"), (-1, "asks")):
            levels = dict(frame[name])
            for price in self.depth_prices[side] - levels.keys():
                self.call("depth", side, price, 0)
            for price, qty in levels.items():
                self.call("depth", side, price, qty)
            self.depth_prices[side] = set(levels)

    @property
    def stats(self):
        value = Stats()
        self.call("stats", C.byref(value))
        return value

    def fills_since(self, index):
        result = []
        for i in range(index, self.stats.fills):
            fill = Fill()
            self.call("fill", i, C.byref(fill))
            result.append({k: getattr(fill, k) for k, _ in fill._fields_})
        return result

    def quote(self, timestamp_ns, mid, policy, sigma):
        e = self.execution
        return self.call("quote", timestamp_ns, mid, policy.half_spread, policy.skew, policy.inventory_band,
                         e["quote_size"], sigma, e["max_volatility_ticks"], e["max_loss"], e["max_drawdown"], policy.version)

    def flatten(self, timestamp_ns):
        self.call("flatten", timestamp_ns)
