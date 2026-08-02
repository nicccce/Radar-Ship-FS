"""Bounded replay of joint environment transitions."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from radar_ship_fs.rl.transition import JointTransition


class JointReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[JointTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: JointTransition) -> None:
        if not transition.applied:
            raise ValueError("rejected/no-op transitions must not enter stable replay")
        self._items.append(transition)

    def sample(self, batch_size: int, rng) -> list[JointTransition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        take = min(int(batch_size), len(self._items))
        if take == 0:
            return []
        indices = rng.numpy.choice(len(self._items), size=take, replace=False)
        items = tuple(self._items)
        return [items[int(index)] for index in indices]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "items": list(self._items)}

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("checkpoint replay capacity does not match current configuration")
        self._items.clear()
        self._items.extend(state["items"])

    def __iter__(self) -> Iterable[JointTransition]:
        return iter(tuple(self._items))
