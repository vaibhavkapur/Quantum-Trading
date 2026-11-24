import hashlib
import io
import json
from pathlib import Path

import numpy as np

from scenarios import correlated_returns
from .config import fingerprint


def dataset_spec(config):
    return {"format_version": 1, "generator": "numpy-pcg64-eigenfactor-einsum-v1", "seed": config.seed,
            "assets": config.portfolio.assets, "model": config.model.model_dump(),
            "observations": config.data.observations, "start_date": config.data.start_date}


def manifest_path(config, root):
    return Path(root) / config.data.directory / (fingerprint(dataset_spec(config))[:20] + ".manifest.json")


def prepare_dataset(config, root):
    path = manifest_path(config, root)
    if path.exists():
        load_dataset(config, root)
        return path
    model = config.model
    returns = correlated_returns(model.mean, model.covariance, config.data.observations, config.seed,
                                 model.distribution, model.df)
    start = np.datetime64(config.data.start_date, "D")
    if np.isnat(start):
        raise ValueError("start date must be a valid ISO date")
    dates = np.busday_offset(start, np.arange(len(returns)), roll="forward")
    stream = io.StringIO()
    stream.write("timestamp," + ",".join(config.portfolio.assets) + "\n")
    for date, row in zip(dates, returns):
        stream.write(str(date) + "T00:00:00Z," + ",".join(format(float(x), ".17g") for x in row) + "\n")
    content = stream.getvalue().encode()
    checksum = hashlib.sha256(content).hexdigest()
    csv_path = path.parent / (checksum + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != checksum:
            raise ValueError("immutable dataset checksum mismatch")
    else:
        with csv_path.open("xb") as handle:
            handle.write(content)
    manifest = {"spec": dataset_spec(config), "spec_sha256": fingerprint(dataset_spec(config)),
                "file": csv_path.name, "sha256": checksum, "license": "CC0-1.0 (generated synthetic data)",
                "timestamp_normalization": "UTC midnight, synthetic weekdays, no exchange holiday calendar",
                "units": "one-day simple returns; not market observations"}
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return path


def load_dataset(config, root):
    path = manifest_path(config, root)
    if not path.exists():
        raise FileNotFoundError("dataset missing; run python -m project.prepare_data --config configs/demo.yaml first")
    manifest = json.loads(path.read_text())
    expected = dataset_spec(config)
    if manifest["spec"] != expected or manifest["spec_sha256"] != fingerprint(expected):
        raise ValueError("dataset specification fingerprint mismatch")
    if manifest["file"] != manifest["sha256"] + ".csv":
        raise ValueError("invalid content-addressed dataset name")
    content = (path.parent / manifest["file"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("immutable dataset checksum mismatch")
    text = content.decode()
    if text.splitlines()[0] != "timestamp," + ",".join(config.portfolio.assets):
        raise ValueError("dataset asset order does not match configuration")
    returns = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1, usecols=range(1, 1+len(config.portfolio.assets)), ndmin=2)
    if returns.shape != (config.data.observations, len(config.portfolio.assets)) or not np.isfinite(returns).all():
        raise ValueError("invalid dataset dimensions or values")
    return returns, manifest
