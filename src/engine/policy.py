"""Value policy (COMP-005, forward path) — a per-agent value network with ε-greedy selection.

Each feature-agent (TASK-208) holds one :class:`ValuePolicy`: a small value network that maps the
agent's fixed-length state vector to an estimated value for each of its two actions —
``ACTION_DESELECT`` and ``ACTION_SELECT`` — and selects between them ε-greedily (greedy by value
most of the time, random the rest). This module is the *acting* half of the engine; the temporal-
difference update that trains the network from stored experience is TASK-209, which drives the
update-ready head left exposed here via :meth:`ValuePolicy.update` and
:meth:`ValuePolicy.parameters`.

The network is a PyTorch ``nn.Module`` head with the reference architecture (two 128-unit ReLU
hidden layers, ASM-002/ASM-005): ``Linear(state_dim, 128) → ReLU → Linear(128, 128) → ReLU →
Linear(128, 2)``, its shape and activation configured entirely from primitives the caller passes —
this module imports neither :mod:`config` nor :mod:`rng`, so it stays a thin, reusable policy over
whatever state dimensionality the seam supplies. Re-expressing the value head on the automatic-
differentiation framework (replacing the prior scikit-learn ``MLPRegressor``) lets a single learning
signal later flow back across the encoder→value boundary (TASK-005); here the encoder stays fixed
and each head owns a persistent ``Adam`` optimizer mirroring ``MLPRegressor``'s internal Adam state.

Determinism (CON-003 / CON-R-001): the head's weight initialization is seeded by the
``random_state`` integer the caller draws from the single shared RNG — sourced from a *local*
``torch.Generator`` so it never disturbs the global torch RNG — and ε-greedy action selection draws
its randomness from that same shared RNG. So a given seed reproduces identical initial parameters
and identical actions.

Satisfies COMP-005 (value-network forward path + ε-greedy) -> REQ-005 (policy/action portion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    # Annotation-only: the policy receives the shared RNG as a parameter at selection time but
    # does not depend on the rng module structurally (kept thin and import-light, per plan).
    from rng import SeededRng

# The two actions every feature-agent chooses between for its own feature.
ACTION_DESELECT = 0
ACTION_SELECT = 1
N_ACTIONS = 2

# CPU reference platform only (CPU-only Non-Goal): heads are pinned to CPU so runs are reproducible
# and free of GPU/MPS nondeterminism.
_DEVICE = torch.device("cpu")

# Activation name (config.activation) -> the torch module that realizes it. Kept explicit so an
# unsupported activation fails loudly rather than silently mis-building the head.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
}


def _build_head(
    state_dim: int,
    hidden_layer_sizes: Sequence[int],
    activation: str,
) -> nn.Sequential:
    """Build the value head ``Linear → activation → … → Linear(*, N_ACTIONS)``.

    Mirrors the reference two-128-unit-ReLU shape (ASM-002): one ``Linear`` + activation per hidden
    width, then a final ``Linear`` to the length-:data:`N_ACTIONS` value vector. Built on CPU.
    """
    try:
        activation_cls = _ACTIVATIONS[activation]
    except KeyError as exc:  # pragma: no cover - configuration guard
        raise ValueError(
            f"unsupported activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}"
        ) from exc

    layers: list[nn.Module] = []
    in_features = int(state_dim)
    for width in hidden_layer_sizes:
        layers.append(nn.Linear(in_features, int(width)))
        layers.append(activation_cls())
        in_features = int(width)
    layers.append(nn.Linear(in_features, N_ACTIONS))
    return nn.Sequential(*layers).to(_DEVICE)


def _seed_head(head: nn.Sequential, random_state: int) -> None:
    """Deterministically (re)initialize ``head``'s parameters from a per-feature seed.

    Newer PyTorch releases let init functions consume a local :class:`torch.Generator`. The
    ``dl-lab`` environment currently carries an older torch that lacks that keyword, so the fallback
    temporarily seeds and restores the CPU RNG to preserve the no-global-RNG-side-effect contract.
    """

    def _init(generator: torch.Generator | None) -> None:
        with torch.no_grad():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    kwargs = {"generator": generator} if generator is not None else {}
                    nn.init.kaiming_uniform_(module.weight, a=5**0.5, **kwargs)
                    if module.bias is not None:
                        fan_in = module.weight.shape[1]
                        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
                        nn.init.uniform_(module.bias, -bound, bound, **kwargs)

    generator = torch.Generator(device=_DEVICE)
    generator.manual_seed(int(random_state))
    try:
        _init(generator)
    except TypeError as exc:
        if "generator" not in str(exc):
            raise
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(random_state))
            _init(None)
        finally:
            torch.random.set_rng_state(state)


class ValuePolicy:
    """A value network over a fixed-length state, with ε-greedy action selection.

    Predicts a length-:data:`N_ACTIONS` value vector ``[value(DESELECT), value(SELECT)]`` for a
    given state and picks an action ε-greedily. The underlying PyTorch head is trained through
    :meth:`update` (driven by the temporal-difference updater, TASK-209) and its learnable
    parameters are exposed via :meth:`parameters` for a future joint optimizer (TASK-005).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: str,
        learning_rate: float,
        random_state: int,
    ) -> None:
        self.state_dim = int(state_dim)
        # Differentiable value head on the autodiff framework, CPU-pinned, seeded deterministically
        # from the per-feature random_state (no global-RNG side effect). Unlike the prior unfitted
        # MLPRegressor, a freshly-initialized head can predict immediately, so no init "partial_fit"
        # on a neutral sample is needed.
        self.head = _build_head(self.state_dim, hidden_layer_sizes, activation)
        _seed_head(self.head, random_state)
        # Persistent per-head Adam (Adam state accumulates across updates, mirroring MLPRegressor's
        # internal Adam); lr from config.learning_rate, default betas 0.9/0.999.
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.MSELoss()

    def parameters(self):
        """Iterate the head's learnable parameters (for a future joint optimizer, TASK-005)."""
        return self.head.parameters()

    def values(self, state: np.ndarray) -> np.ndarray:
        """Return the length-:data:`N_ACTIONS` estimated value vector for ``state``.

        Forward-only (no grad), returning a plain numpy vector — the output contract is unchanged
        from the prior ``MLPRegressor.predict`` path.
        """
        state_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, self.state_dim), device=_DEVICE
        )
        with torch.no_grad():
            out = self.head(state_t)
        return out.cpu().numpy().reshape(N_ACTIONS).astype(float)

    def update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Apply one Adam step minimizing MSE between the head's outputs and ``targets``.

        ``states`` is ``(batch, state_dim)`` and ``targets`` is ``(batch, N_ACTIONS)``. The per-head
        loss (scalar MSE over the 2-vector) is computed, back-propagated, and one persistent-Adam
        step is taken; the loss value is returned (per-head loss, for visibility / a future joint
        learner).
        """
        states_t = torch.as_tensor(
            np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim), device=_DEVICE
        )
        targets_t = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).reshape(-1, N_ACTIONS), device=_DEVICE
        )
        self.optimizer.zero_grad()
        predictions = self.head(states_t)
        loss = self.loss_fn(predictions, targets_t)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def select_action(
        self,
        state: np.ndarray,
        rng: "SeededRng",
        exploitation_probability: float,
    ) -> int:
        """Ε-greedily pick an action for ``state`` using the shared ``rng``.

        With probability ``exploitation_probability`` the greedy action (the higher-valued one) is
        chosen; otherwise a random action is drawn uniformly over both. All randomness comes from
        the shared ``rng`` (CON-003), so selection is reproducible under the seed. Greedy ties break
        toward the lower action index (``argmax`` convention).
        """
        if rng.numpy.random() < exploitation_probability:
            return int(np.argmax(self.values(state)))
        return int(rng.numpy.integers(0, N_ACTIONS))
