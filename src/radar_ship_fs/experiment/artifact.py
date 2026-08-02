"""One public artifact service for stable experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch

from radar_ship_fs import __version__


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *command], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def development_fingerprint(context) -> str:
    """Hash search-visible data plus held-out row identity, never held-out feature values."""
    digest = hashlib.sha256()
    for value in (
        context.split.train.X,
        context.split.train.y,
        context.split.train.indices,
        context.split.test.indices,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class ArtifactStore:
    """Atomic JSON/CSV output with manifest/config collision protection."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.selection_path = self.run_dir / "selection.json"
        self.training_path = self.run_dir / "training.csv"
        self.training_jsonl_path = self.run_dir / "training.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.pt"

    @staticmethod
    def write_json(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)

    @staticmethod
    def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
        rows = list(rows)
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @classmethod
    def prepare_root(cls, root: str | Path, spec) -> None:
        """Reserve one result root for exactly one algorithm/config identity."""
        root = Path(root)
        marker = root / "experiment-root.json"
        expected = {
            "artifact_schema_version": 1,
            "algorithm_version": spec.algorithm_version,
            "config_hash": spec.config_hash,
        }
        if marker.is_file():
            with marker.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != expected:
                raise ValueError(
                    f"result root is already reserved for a different algorithm or config: {root}"
                )
            return
        if root.is_dir() and any(root.iterdir()):
            raise ValueError(
                f"refusing to mix stable artifacts into an existing unversioned result root: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)
        cls.write_json(expected, marker)

    def existing_selection(self, identity: dict, expected_steps: int) -> dict | None:
        if not self.selection_path.is_file():
            return None
        with self.selection_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("experiment_signature") != _jsonable(identity):
            return None
        if len(payload.get("trajectory", [])) != expected_steps:
            return None
        if not payload.get("selected_clean_indices"):
            return None
        return payload

    def write_manifest(self, *, spec, method, seed: int, context, identity: dict) -> None:
        if self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("experiment_signature") != _jsonable(identity):
                raise ValueError(f"output directory already contains a different experiment: {self.run_dir}")
            return
        metadata = context.split.train.metadata or {}
        labels, counts = np.unique(context.split.train.y, return_counts=True)
        manifest = {
            "artifact_schema_version": 1,
            "package_version": __version__,
            "experiment_signature": identity,
            "config": spec.canonical_dict(),
            "method": asdict(method),
            "seed": seed,
            "dataset_metadata": metadata,
            "data_summary": {
                "search_rows": int(context.split.train.X.shape[0]),
                "held_out_rows": int(context.split.test.X.shape[0]),
                "n_features": int(context.n_features),
                "class_counts": {str(label): int(count) for label, count in zip(labels, counts)},
                "development_fingerprint": identity["development_fingerprint"],
            },
            "runtime": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": str(torch.__version__),
                "device": "cpu",
            },
            "git": {
                "commit": _git(["rev-parse", "HEAD"]),
                "dirty": bool(_git(["status", "--porcelain"])),
            },
        }
        self.write_json(manifest, self.manifest_path)

    def write_result(self, result, *, identity: dict, context) -> dict:
        metadata = context.split.train.metadata or {}
        original_ids = metadata.get("final_feature_ids", list(range(context.n_features)))
        selected = tuple(int(value) for value in result.selection.selected)
        metrics = [metric.as_dict() for metric in result.metrics]
        payload = {
            "artifact_schema_version": 1,
            "algorithm_version": "stable_v1",
            "experiment_signature": identity,
            "selected_clean_indices": list(selected),
            "selected_original_feature_ids": [int(original_ids[index]) for index in selected],
            "selected_count": len(selected),
            "best_dt_inner_cv_accuracy": (
                result.metrics[-1].best_accuracy if result.metrics else result.initial_accuracy
            ),
            "initial_candidate": {
                "selected_clean_indices": list(result.initial_subset),
                "accuracy": result.initial_accuracy,
            },
            "learner_updates": result.learner_updates,
            "rejected_transitions": result.rejected_transitions,
            "trajectory": metrics,
        }
        self.write_json(payload, self.selection_path)
        csv_rows = []
        for row in metrics:
            flattened = dict(row)
            flattened["subset"] = ";".join(str(value) for value in row["subset"])
            csv_rows.append(flattened)
        self.write_csv(csv_rows, self.training_path)
        return payload


class ArtifactObserver:
    """Persist immutable step events without reaching into session or trainer internals."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self._rows: list[dict[str, Any]] = []
        if store.training_jsonl_path.is_file():
            with store.training_jsonl_path.open("r", encoding="utf-8") as handle:
                self._rows = [json.loads(line) for line in handle if line.strip()]

    def _write_jsonl(self) -> None:
        path = self.store.training_jsonl_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _csv_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened = []
        for row in rows:
            item = dict(row)
            item["subset"] = ";".join(str(value) for value in row["subset"])
            flattened.append(item)
        return flattened

    def on_step(self, event) -> None:
        row = event.metrics.as_dict()
        step = int(row["step"])
        self._rows = [existing for existing in self._rows if int(existing["step"]) < step]
        self._rows.append(row)
        self._write_jsonl()

    def on_update(self, event) -> None:
        return None

    def on_checkpoint(self, event) -> None:
        self.store.write_csv(self._csv_rows(self._rows), self.store.training_path)
