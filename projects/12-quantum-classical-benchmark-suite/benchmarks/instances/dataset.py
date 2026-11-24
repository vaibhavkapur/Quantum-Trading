from __future__ import annotations

import json
from pathlib import Path

from .schema import canonical_json, digest, validate_instance


def write_immutable(path: Path, value: dict) -> None:
    content = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as out:
            out.write(content)
    except FileExistsError:
        if path.read_text() != content:
            raise ValueError(f"immutable dataset conflict: {path}; use a new dataset directory")


def prepare(config: dict, base: Path) -> Path:
    from .generators import GENERATOR_VERSION, generate

    root = base / config["dataset"]
    entries = []
    # Derivation is independent of iteration order and PYTHONHASHSEED.
    for spec in config["instances"]:
        for size in spec["sizes"]:
            seed = int(digest([config["data_seed"], spec["track"], size])[:8], 16)
            instance = generate(spec["track"], size, seed)
            filename = instance["instance_id"] + ".json"
            write_immutable(root / filename, instance)
            entries.append({"file": filename, "sha256": digest(instance),
                            "instance_id": instance["instance_id"], "track": spec["track"], "size": size})
    manifest = {"schema_version": 1, "generator": GENERATOR_VERSION, "license": "CC0-1.0",
                "provenance": "Deterministic synthetic data; no external dataset or market timestamps.",
                "timestamp_policy": "No time-series observations; experiment timestamps use UTC ISO-8601.",
                "data_seed": config["data_seed"], "instances": sorted(entries, key=lambda e: e["instance_id"])}
    write_immutable(root / "manifest.json", manifest)
    return root / "manifest.json"


def load_manifest(path: Path, config: dict | None = None) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("instances"):
        raise ValueError("invalid dataset manifest")
    files = [entry["file"] for entry in manifest["instances"]]
    if len(files) != len(set(files)):
        raise ValueError("duplicate dataset entries")
    for entry in manifest["instances"]:
        target = (path.parent / entry["file"]).resolve()
        if target.parent != path.parent.resolve():
            raise ValueError("dataset files must be directly inside the dataset directory")
        instance = load_instance(target, entry["sha256"])
        if any(instance["problem"][key] != entry[key] for key in ("track", "size")) or instance["instance_id"] != entry["instance_id"]:
            raise ValueError("dataset entry metadata mismatch")
    if config is not None:
        expected = {(s["track"], n) for s in config["instances"] for n in s["sizes"]}
        actual = {(e["track"], e["size"]) for e in manifest["instances"]}
        if expected != actual or config["data_seed"] != manifest["data_seed"]:
            raise ValueError("dataset does not match config; prepare a new versioned directory")
    return manifest


def load_instance(path: Path, expected_digest: str) -> dict:
    instance = json.loads(path.read_text())
    if digest(instance) != expected_digest:
        raise ValueError(f"dataset checksum mismatch: {path.name}")
    validate_instance(instance)
    return instance
