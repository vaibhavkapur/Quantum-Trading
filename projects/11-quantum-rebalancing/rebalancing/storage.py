"""Content-addressed run files and an append-only, hash-chained SQLite audit log."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .model import canonical, digest


class RunStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "audit.sqlite3"
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS runs (sequence INTEGER PRIMARY KEY, run_id TEXT UNIQUE NOT NULL, recommendation_id TEXT NOT NULL, created_at TEXT NOT NULL, previous_hash TEXT NOT NULL, entry_hash TEXT NOT NULL, payload TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def save(self, result: dict) -> str:
        run_id = digest(result)
        payload = canonical(result)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is None:
                previous = connection.execute("SELECT entry_hash FROM runs ORDER BY sequence DESC LIMIT 1").fetchone()
                previous_hash = previous[0] if previous else "0" * 64
                created_at = datetime.now(timezone.utc).isoformat()
                entry_hash = digest({"previous_hash": previous_hash, "run_id": run_id,
                                     "recommendation_id": result["recommendation_id"], "created_at": created_at})
                connection.execute("INSERT INTO runs (run_id,recommendation_id,created_at,previous_hash,entry_hash,payload) VALUES (?,?,?,?,?,?)",
                                   (run_id, result["recommendation_id"], created_at, previous_hash, entry_hash, payload))
        # SQLite is authoritative; a retried save also repairs a missing export.
        path = self.directory / f"{run_id}.json"
        if path.exists():
            if path.read_text() != payload + "\n":
                raise ValueError(f"immutable run file has changed: {path}")
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            except FileExistsError:
                # Another writer can export the same committed record first.
                if path.read_text() != payload + "\n":
                    raise ValueError(f"immutable run file has changed: {path}") from None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return run_id

    def get(self, run_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def verify(self) -> dict:
        previous, count = "0" * 64, 0
        with self.connect() as connection:
            for run_id, recommendation, created_at, previous_hash, entry_hash, payload in connection.execute(
                    "SELECT run_id,recommendation_id,created_at,previous_hash,entry_hash,payload FROM runs ORDER BY sequence"):
                expected = digest({"previous_hash": previous, "run_id": run_id,
                                   "recommendation_id": recommendation, "created_at": created_at})
                try:
                    parsed = json.loads(payload)
                    valid_payload = digest(parsed) == run_id and parsed["recommendation_id"] == recommendation
                except (ValueError, KeyError, TypeError):
                    valid_payload = False
                if previous_hash != previous or entry_hash != expected or not valid_payload:
                    return {"valid": False, "entries_checked": count, "failed_run_id": run_id}
                previous, count = entry_hash, count + 1
        return {"valid": True, "entries_checked": count, "head_hash": previous}

