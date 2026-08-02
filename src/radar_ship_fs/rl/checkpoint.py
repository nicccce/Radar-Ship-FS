"""Atomic, versioned stable-training checkpoints."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch

_CHECKPOINT_SCHEMA = 1


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        envelope = {"schema_version": _CHECKPOINT_SCHEMA, "payload": payload}
        torch.save(envelope, temporary)
        os.replace(temporary, self.path)

    def load(self) -> dict:
        try:
            envelope = torch.load(self.path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch 1.x has no weights_only argument.
            envelope = torch.load(self.path, map_location="cpu")
        if envelope.get("schema_version") != _CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        return envelope["payload"]


def capture_rng_state(rng) -> dict:
    return {
        "numpy": copy.deepcopy(rng.numpy.bit_generator.state),
        "python": rng.python.getstate(),
        "torch": torch.random.get_rng_state().clone(),
    }


def restore_rng_state(rng, state: dict) -> None:
    rng.numpy.bit_generator.state = copy.deepcopy(state["numpy"])
    rng.python.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
