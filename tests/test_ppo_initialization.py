"""PPO initialization strategy tests."""

from dataclasses import replace

import numpy as np
import pytest

from radar_ship_fs.experiment.config import load_experiment_spec
from radar_ship_fs.ppo.run_session import _random_initial_mask


def test_random_initial_mask_is_seeded_and_budget_exact() -> None:
    first = _random_initial_mask(65, 32, 42)
    repeated = _random_initial_mask(65, 32, 42)
    different = _random_initial_mask(65, 32, 43)

    assert first.dtype == np.bool_
    assert int(first.sum()) == 32
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize(("n_features", "budget"), [(0, 1), (5, 0), (5, 6)])
def test_random_initial_mask_rejects_invalid_budget(n_features: int, budget: int) -> None:
    with pytest.raises(ValueError, match="feature_budget"):
        _random_initial_mask(n_features, budget, 42)


def test_random_initialization_is_configured_and_validated() -> None:
    spec = load_experiment_spec("configs/v16n/run_ppo_random_k32.toml")

    assert spec.ppo.initialization == "random"
    with pytest.raises(ValueError, match="ppo.initialization"):
        replace(spec, ppo=replace(spec.ppo, initialization="unknown")).validate()
