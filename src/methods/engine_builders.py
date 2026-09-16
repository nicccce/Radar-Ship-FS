"""IRFS engine builders (COMP-011) — assemble reinforced engines over the shared IRFS state/reward.

Selects the state encoder behind the engine's unchanged ``StateEncoder`` seam (TASK-003 / COMP-004 /
DEC-005) and binds it with the personalized reward into a :class:`~engine.explore.ReinforcedEngine`.
Every reinforced configuration shares one state representation and one personalized reward and differs
*only* by the injected advisor; the no-trainer configuration passes ``advisor=None``. This is the
composition point where the richer IRFS feedback (COMP-007 tree state, COMP-008/009 personalized reward)
drops into PHASE-002's proven engine behind the *unchanged* interface and subset contract (DEC-004,
CON-002 / COMPAT-001).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from engine.explore import ReinforcedEngine
from methods.seam_adapters import (
    PersonalizedRewardSeamAdapter,
    TrainableGCNSeamAdapter,
    TreeStateSeamAdapter,
)
from rng import SeededRng
from state.encoder import TreeStructuredStateEncoder

if TYPE_CHECKING:
    from config import IrfsConfig
    from engine.seam import ActionAdvisor
    from harness.contract import SelectionContext, SubsetSelection


def select_state_encoder(
    config: "Optional[IrfsConfig]" = None,
    *,
    encoder: Optional[TreeStructuredStateEncoder] = None,
    rng: Optional[SeededRng] = None,
) -> "tuple[object, Optional[object]]":
    """The encoder selector (TASK-003 / COMP-004 / DEC-005): pick the seam adapter from ``config``.

    Returns ``(seam_adapter, trainable_params_or_None)`` so the engine builder drops the chosen encoder
    in behind the **unchanged** ``StateEncoder`` seam (CON-R-002) and a caller can hand a selected trained
    encoder's parameters to the future joint optimizer (TASK-005). This is the explicit, observable
    registration handoff RISK-003 demands — not implied by construction:

    - ``config is None`` or ``config.state_encoder == "fixed"`` → ``(TreeStateSeamAdapter(encoder), None)``:
      PHASE-004's fixed-weight encoder, byte-identical to today's baseline (AC-007); no trainable params.
    - ``config.state_encoder == "trained_gcn"`` → ``(TrainableGCNSeamAdapter(config, rng=rng), params)``:
      the learnable encoder (TASK-002), with ``params`` its ``parameters()`` iterable when an ``rng`` is
      supplied for eager registration, else the adapter's ``parameters`` accessor (resolves after the first
      encode seeds it from ``context.rng``). Either way the params are observable, never silently inert.

    Unknown ``state_encoder`` values raise ``ValueError`` rather than silently falling back to fixed.
    """
    choice = "fixed" if config is None else config.state_encoder
    if choice == "fixed":
        return TreeStateSeamAdapter(encoder), None
    if choice == "trained_gcn":
        assert config is not None  # only reachable when config supplied (choice read from it)
        adapter = TrainableGCNSeamAdapter(config, rng=rng)
        # Eager params when an rng built the encoder now; otherwise expose the accessor
        # (resolves post-encode).
        params = adapter.parameters() if rng is not None else adapter.parameters
        return adapter, params
    raise ValueError(f"Unknown state_encoder {choice!r}; expected one of ('fixed', 'trained_gcn')")


def build_advised_engine(
    advisor: "Optional[ActionAdvisor]" = None,
    encoder: Optional[TreeStructuredStateEncoder] = None,
    *,
    config: "Optional[IrfsConfig]" = None,
) -> ReinforcedEngine:
    """Build a reinforced engine over the shared IRFS state/reward, optionally bound to an advisor.

    Every reinforced configuration shares one state representation and one personalized reward and
    differs *only* by ``advisor`` (TASK-411/412): ``None`` is the no-trainer configuration (the
    engine skips the advice seam entirely, so its behavior is unchanged — TASK-406); a trainer or
    hybrid advisor (``methods.advice``) overrides hesitant agents' votes each step. ``encoder`` lets
    a caller override the wrapped tree-structured encoder (e.g. for tests). ``config`` selects which
    encoder drops in behind the seam (TASK-003): ``None`` or ``state_encoder="fixed"`` keeps the
    fixed-weight adapter byte-identical to the baseline (AC-007), ``"trained_gcn"`` selects the
    trainable encoder. Fresh adapter instances are created per call so each engine carries its own
    memoization/frequency state.

    The selected encoder's trainable parameters (if any) are discarded by this builder, which
    returns only the engine; :func:`build_advised_engine_with_registration` returns them too for the
    TASK-005 optimizer.
    """
    seam_adapter, _params = select_state_encoder(config, encoder=encoder)
    return ReinforcedEngine(
        encoder=seam_adapter,
        reward=PersonalizedRewardSeamAdapter(),
        advisor=advisor,
    )


def build_advised_engine_with_registration(
    advisor: "Optional[ActionAdvisor]" = None,
    encoder: Optional[TreeStructuredStateEncoder] = None,
    *,
    config: "Optional[IrfsConfig]" = None,
    rng: Optional[SeededRng] = None,
) -> "tuple[ReinforcedEngine, Optional[object]]":
    """Like :func:`build_advised_engine` but also returns the selected encoder's trainable params.

    Returns ``(engine, trainable_params_or_None)`` — the explicit registration handoff (RISK-003)
    that PHASE-002's joint optimizer (TASK-005) consumes to own ``encoder ∪ heads``. ``None`` for
    the fixed encoder (inert registration, as designed). When ``rng`` is supplied with a
    ``trained_gcn`` config the trained encoder is built and its ``parameters()`` are registered
    eagerly at build time; otherwise the params accessor is returned and resolves once the first
    encode seeds the encoder from ``context.rng``. The engine itself is identical to
    :func:`build_advised_engine` — same seam, same reward, same advisor.
    """
    seam_adapter, params = select_state_encoder(config, encoder=encoder, rng=rng)
    engine = ReinforcedEngine(
        encoder=seam_adapter,
        reward=PersonalizedRewardSeamAdapter(),
        advisor=advisor,
    )
    return engine, params


def build_no_trainer_engine(
    encoder: Optional[TreeStructuredStateEncoder] = None,
) -> ReinforcedEngine:
    """Build the no-trainer reinforced engine — :func:`build_advised_engine` with no advisor.

    Retained as the named no-trainer entry point (TASK-405/406 consumers); behavior is identical to
    the pre-advice engine because no advisor is injected.
    """
    return build_advised_engine(advisor=None, encoder=encoder)


def run_no_trainer(context: "SelectionContext") -> "SubsetSelection":
    """Run the no-trainer IRFS configuration end-to-end, returning the best subset + per-step
    series."""
    return build_no_trainer_engine().select(context)
