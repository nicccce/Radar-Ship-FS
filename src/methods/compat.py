from dataclasses import dataclass
from typing import Sequence

@dataclass
class SubsetSelection:
    selected: tuple[int, ...]

def make_selection(selected: Sequence[int]) -> SubsetSelection:
    return SubsetSelection(tuple(int(x) for x in selected))
