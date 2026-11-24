"""Versioned synthetic L2 frames, normalized to integer nanoseconds and ticks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from .config import digest

SCHEMA = "mm-l2-v1"


def generate(config, seed, scenario):
    d = config["data"]
    rng = np.random.default_rng(np.random.SeedSequence([d["seed"], seed]))
    mid = d["initial_mid_ticks"]
    test_start = int(d["steps"] * d["train_fraction"]) + int(d["steps"] * d["validation_fraction"])
    excitation = 0.0
    frames = []
    for i in range(d["steps"]):
        regime = scenario if i >= test_start else "normal"
        sigma = 2.2 if regime == "volatile" else (1.2 if regime == "liquidity_crisis" else 0.6)
        drift = 0.35 if regime == "trend" else 0.0
        move = int(np.rint(rng.normal(drift, sigma)))
        mid = max(100, min(999000000, mid + move))
        excitation = min(10.0, excitation * 0.65)
        intensity = 2.5 + excitation
        buy = int(rng.poisson(intensity + max(move, 0) * 4))
        sell = int(rng.poisson(intensity + max(-move, 0) * 4))
        excitation += (buy + sell) * 0.04
        depth = max(1, d["depth_qty"] // 4) if regime == "liquidity_crisis" else d["depth_qty"]
        spread = 3 if regime == "liquidity_crisis" else 1
        bids, asks = [], []
        for level in range(d["depth_levels"]):
            bids.append([mid - spread - level, max(1, int(rng.poisson(depth)))])
            asks.append([mid + spread + level, max(1, int(rng.poisson(depth)))])
        frames.append(dict(timestamp_ns=i*d["interval_ns"], mid_ticks=mid, bids=bids, asks=asks,
                           market_buy_qty=buy, market_sell_qty=sell, buy_first=bool(rng.integers(2)), regime=regime))
    return frames


def validate_frames(frames):
    if not frames:
        raise ValueError("empty replay")
    previous = -1
    for frame in frames:
        timestamp = frame["timestamp_ns"]
        if type(timestamp) is not int or timestamp <= previous or timestamp < 0 or timestamp >= 2**63:
            raise ValueError("timestamps must be strictly increasing int64 nanoseconds")
        previous = timestamp
        mid = frame["mid_ticks"]
        if type(mid) is not int or not 1 < mid <= 999000000:
            raise ValueError("invalid mid price")
        for key in ("market_buy_qty", "market_sell_qty"):
            if type(frame[key]) is not int or not 0 <= frame[key] <= 1000000:
                raise ValueError("invalid market quantity")
        if type(frame["buy_first"]) is not bool:
            raise ValueError("buy_first must be boolean")
        for side in ("bids", "asks"):
            levels = frame[side]
            if not levels or len({p for p, q in levels}) != len(levels):
                raise ValueError("empty or duplicate depth")
            if any(type(p) is not int or type(q) is not int or not 0 < p <= 1000000000 or not 0 < q <= 1000000 for p, q in levels):
                raise ValueError("invalid depth level")
        if not max(p for p,q in frame["bids"]) < mid < min(p for p,q in frame["asks"]):
            raise ValueError("crossed book or mid outside spread")
    return frames


def prepare(config, root="data"):
    spec = {"schema": SCHEMA, "data": config["data"], "seeds": config["seeds"], "scenarios": config["scenarios"]}
    directory = Path(root) / digest(spec)[:16]
    if directory.exists():
        load_dataset(directory)  # Validate hashes, never silently overwrite datasets.
        return directory
    directory.mkdir(parents=True)
    files = []
    for seed in config["seeds"]:
        for scenario in config["scenarios"]:
            frames = validate_frames(generate(config, seed, scenario))
            path = directory / f"{scenario}-{seed}.jsonl"
            contents = "".join(json.dumps(f, separators=(",", ":")) + "\n" for f in frames).encode()
            path.write_bytes(contents)
            files.append(dict(file=path.name, seed=seed, scenario=scenario, sha256=hashlib.sha256(contents).hexdigest(), rows=len(frames)))
    d = config["data"]
    train_end = int(d["steps"] * d["train_fraction"])
    validation_end = train_end + int(d["steps"] * d["validation_fraction"])
    manifest = dict(spec=spec, dataset_id=digest(spec), license="CC0-1.0: generated synthetic data; no exchange data included",
                    timestamps="integer nanoseconds since synthetic epoch, UTC; strictly increasing",
                    splits={"train": [0, train_end], "validation": [train_end, validation_end], "test": [validation_end, d["steps"]]},
                    queue_model="L2 inferred FIFO: new quantity joins tail, reductions cancel newest external quantity first",
                    files=files)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_id"] != digest(manifest["spec"]) or manifest["spec"]["schema"] != SCHEMA:
        raise ValueError("invalid dataset manifest")
    result = {}
    for item in manifest["files"]:
        path = (directory / item["file"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("dataset file must be within dataset directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {item['file']}")
        frames = validate_frames([json.loads(line) for line in raw.splitlines()])
        if len(frames) != item["rows"] or len(frames) != manifest["spec"]["data"]["steps"]:
            raise ValueError("dataset row count mismatch")
        key = (item["scenario"], item["seed"])
        if key in result:
            raise ValueError("duplicate dataset scenario/seed")
        result[key] = frames
    expected = {(s, seed) for s in manifest["spec"]["scenarios"] for seed in manifest["spec"]["seeds"]}
    if set(result) != expected:
        raise ValueError("incomplete dataset")
    return manifest, result
