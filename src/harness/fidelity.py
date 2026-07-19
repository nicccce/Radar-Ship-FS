"""Fidelity notes recorder (COMP-024) — the machine-readable caveat register.

Every place this reproduction filled a gap the reference left open, or deliberately departed from
it, is already explained in prose at its source (the relevant module docstring/comment). This module
does not *decide* any of those choices — they exist upstream — it only **collects** them into one
stable, JSON-serializable register so the run artifact (TASK-504) can carry an auditable list of
caveats instead of leaving them scattered across docstrings. A reader trusts, and can reproduce the
reasoning behind, the headline comparison by reading this block.

Each :class:`FidelityNote` records: a stable ``id``; the ``subject``; the ``gap`` (what the
reference left unspecified, or what we departed from); the ``choice`` we made; the ``rationale``;
and the ``references`` (the REQ/RISK/COMP/AC anchors). The register is a single static tuple —
adding a note is a one-line change here, and nothing recomputes or runs a method to produce it (pure
data).

Values are NOT duplicated here: where a note concerns configurable parameters, it *names* them and
points at the artifact's ``config`` block (which already dumps the whole effective configuration),
so the note cannot drift from the values actually used. The concrete mRMR version likewise stays an
artifact field (TASK-504), not a string baked into the note.

Scope: this module is the COMP-024 data source. Embedding it into the persisted artifact is
TASK-504; referencing it from the run narrative is TASK-505. Nothing here imports a method, the
config, or a metric, so the register stays decoupled and trivially round-trips through JSON.

Satisfies COMP-024 -> REQ-019-B (records RISK-001 / RISK-002 and the dataset-onboarding departures).
"""

from __future__ import annotations

from typing import Any, NamedTuple


class FidelityNote(NamedTuple):
    """One recorded gap-fill or deliberate departure from the reference.

    ``id`` is a stable kebab-case handle (used by readers/tests to find a note); ``gap`` states what
    the reference left unspecified or what was departed from; ``choice`` is what this reproduction
    did; ``rationale`` is why; ``references`` are the spec anchors (REQ/RISK/COMP/AC ids). All
    fields are plain strings (``references`` a string tuple), so a note is JSON-ready by
    construction.
    """

    id: str
    subject: str
    gap: str
    choice: str
    rationale: str
    references: tuple[str, ...]


# --- The recorded register (COMP-024). Add a note by appending one FidelityNote below. ------------
#
# Each note's text is sourced from the in-tree prose that already documents the choice (the cited
# module/line), not re-decided here. Dataset-onboarding notes that only bear on some datasets state
# their applicability in-text (e.g. WDBC is binary and single-subject, so they are inert there).

FIDELITY_NOTES: tuple[FidelityNote, ...] = (
    FidelityNote(
        id="validation-step-protocol",
        subject="Per-step scoring uses the validation partition, not test",
        gap=(
            "The reference's per-step selection protocol is not stated precisely enough to confirm "
            "which partition scores each exploration step."
        ),
        choice=(
            "Every method scores candidate subsets only on split.validation during selection; the "
            "held-out test partition is released exactly once, post-selection, for final reported "
            "metrics (the dual val/test pair, TASK-503)."
        ),
        rationale=(
            "Keeps the leakage invariant intact (no per-step or selection path can read test) while "
            "still reporting an honest generalization number; the validation-vs-test scoring "
            "deviation is the recorded RISK-002 caveat."
        ),
        references=("RISK-002", "REQ-010", "AC-002"),
    ),
    FidelityNote(
        id="unspecified-reference-defaults",
        subject="Reference-unspecified hyperparameters carry pragmatic defaults",
        gap=(
            "The reference does not fix several parameters: the exploration step budget, the "
            "exploration parameter, the correlation-penalty weight beta, the neighbour-vs-global "
            "aggregation mix lambda, the per-node feature set, the state pooling method, and whether "
            "a separate target network is used."
        ),
        choice=(
            "Each is given a recorded default (e.g. the step budget tuned for WDBC convergence; "
            "per-node features = summary statistics [mean, std, min, max]; no separate target "
            "network). The effective value of every one of these is dumped in the artifact's config "
            "block rather than restated here, so this note cannot drift from what was run."
        ),
        rationale=(
            "The choices are disclosed in the effective-configuration view so a reader can see and "
            "reproduce them; they are the accepted RISK-001 mitigation (record, don't hide)."
        ),
        references=("RISK-001", "REQ-012", "AC-009", "Q-003"),
    ),
    FidelityNote(
        id="mrmr-pinned-implementation",
        subject="mRMR baseline is pinned to a specific published implementation",
        gap=(
            "mRMR has several published variants; an unpinned implementation would make the baseline "
            "irreproducible."
        ),
        choice=(
            "The mRMR baseline is pinned to one published implementation; its exact name and version "
            "are captured as a concrete identity field in the run artifact (TASK-504)."
        ),
        rationale=(
            "Pinning the implementation identity makes the classical mRMR number reproducible and "
            "auditable (COMP-007); the version lives in the artifact, not in this note, to avoid two "
            "sources of truth."
        ),
        references=("COMP-007",),
    ),
    FidelityNote(
        id="hybrid-schedule-boundaries",
        subject="Hybrid-teaching switch/withdraw boundaries are a gap-fill",
        gap=(
            "The reference fixes the hybrid-teaching shape (relevance trainer, then DT-importance "
            "trainer, then withdraw guidance) but not the step boundaries between those phases."
        ),
        choice=(
            "The two boundaries default to thirds of the step budget (switch then withdraw), "
            "configurable via hybrid_switch_step / hybrid_withdraw_step (effective values in the "
            "config block)."
        ),
        rationale=(
            "A documented, configurable gap-fill that preserves the reference's three-phase schedule "
            "shape while making the unspecified boundaries explicit."
        ),
        references=("RISK-001", "REQ-004", "COMP-004"),
    ),
    FidelityNote(
        id="full-irfs-equals-hybrid",
        subject="Full IRFS is the Hybrid Teaching configuration",
        gap=(
            "Whether the headline 'full IRFS' is a distinct fifth trainer or the Hybrid Teaching "
            "schedule is a reading of the reference."
        ),
        choice=(
            "full_irfs uses the hybrid (Hybrid Teaching) advisor directly rather than a fabricated "
            "fifth trainer; the redundant standalone 'hybrid' method (byte-identical to full_irfs) "
            "has been removed."
        ),
        rationale=(
            "The reference's headline proposed method is Hybrid Teaching (Sec 3.1.3, Sec 4.5 'IRFS "
            "with HT'), so full IRFS *is* the Hybrid Teaching configuration (reading A, user-confirmed)."
        ),
        references=("REQ-010", "AC-005", "DEC-001"),
    ),
    FidelityNote(
        id="marlfs-faithful-baseline",
        subject="MARLFS baseline uses none of the three IRFS contributions",
        gap=(
            "The reference casts MARLFS both as an external baseline (Sec 4.3/4.4) and as the no-trainer "
            "bottom rung of the trainer study (Sec 4.5), without stating whether it shares the proposed "
            "GCN state and personalized reward."
        ),
        choice=(
            "marlfs is the faithful vanilla multi-agent RL baseline: no trainer, the minimal "
            "[relevance, redundancy] state (not the tree-structured/GCN encoder), and the uniform overall "
            "reward Acc - beta*R (not the personalized per-agent reward). It is a distinct method from "
            "no_trainer, which is IRFS-without-trainer (GCN state + personalized reward). MARLFS therefore "
            "ignores config.state_encoder and computes its reward correlation on the validation partition "
            "(the PHASE-002 substrate)."
        ),
        rationale=(
            "The variant study (Sec 4.6) presents the GCN state (SRDT) and personalized reward (PRS) as "
            "improvements over the MARLFS baseline, so MARLFS must lack both; giving the baseline any of "
            "the three contributions would inflate it and understate the reported gains."
        ),
        references=("REQ-010", "AC-005"),
    ),
    FidelityNote(
        id="per-agent-credit-scheme",
        subject="Per-agent credit assignment and frequency-history interpretation",
        gap=(
            "The reference under-specifies how the overall reward becomes each agent's learning "
            "signal, and what 'historical selection' counts mean for the frequency reward scheme."
        ),
        choice=(
            "Default per_agent_credit='reference' applies r_i = weight_i*(Acc - beta*R) with "
            "deselected agents pinned to zero (the honest reproduction); 'symmetric' (every agent "
            "gets the full overall reward) is an opt-in diagnostic deviation. For the frequency "
            "scheme, the per-feature selection tally is advanced once per committed subset."
        ),
        rationale=(
            "Any non-'reference' credit mode is an opt-in deviation recorded as a fidelity note; the "
            "frequency-count-per-committed-subset rule is a recorded interpretation of the "
            "under-specified 'historical selection'. The headline default (dt_importance) needs no "
            "history."
        ),
        references=("RISK-001", "COMP-024", "Q-001"),
    ),
    FidelityNote(
        id="deselected-agent-state",
        subject="Deselected-agent state uses a once-built shared graph (global term only)",
        gap=(
            "The reference defines the tree-structured state over the selected subset and does not "
            "specify a per-agent state for a deselected feature."
        ),
        choice=(
            "encode_all builds the augmented graph once over the selected set S; a selected agent's "
            "row is bit-identical to a per-feature build, while a deselected agent is attached to that "
            "fixed S graph via its train-correlations to S (the Step-4 global term only, no per-agent "
            "DT refit, no tree-edge neighbours)."
        ),
        rationale=(
            "A deliberate, performance-motivated departure from the earlier per-deselected S-union-{i} "
            "build: it removes the per-deselected recomputation (days to ~an hour on high-dimensional "
            "datasets; small datasets unaffected) while leaving the committed subset and its metrics "
            "unchanged. The richer shared-S-tree representation is held in reserve if it ever changes "
            "convergence materially."
        ),
        references=("RISK-001", "REQ-019-B", "REQ-007"),
    ),
    FidelityNote(
        id="subject-aware-split",
        subject="Group-aware split keeps whole subjects on one side (departs from random 80/20)",
        gap=(
            "The reference uses a plain random 80/20 split, which would let one subject's samples "
            "span train/validation/test on subject-structured datasets."
        ),
        choice=(
            "On grouped datasets the split is group-aware (whole groups, e.g. subjects, stay on one "
            "side) and therefore unstratified. Applies to subject-structured datasets (e.g. Parkinson's); "
            "inert for WDBC, which has no groups and uses the stratified random path unchanged."
        ),
        rationale=(
            "Prevents subject leakage across partitions on grouped datasets; the departure from the "
            "reference's plain random 80/20 is recorded here."
        ),
        references=("REQ-022", "ASM-001", "REQ-019-B"),
    ),
    FidelityNote(
        id="multiclass-l1-ovr",
        subject="L1 baseline wraps One-vs-Rest for multiclass labels",
        gap=(
            "The liblinear L1 solver refuses n_classes >= 3 directly, so a multiclass L1 baseline "
            "needs a defined multiclass strategy the reference does not specify."
        ),
        choice=(
            "The L1 baseline is wrapped in OneVsRestClassifier: each class gets its own binary L1 fit "
            "and a feature survives if it is non-zero for any class. Applies to any dataset with "
            "n_classes >= 3; for a binary label this is a single fit identical to the bare estimator, so "
            "the supported binary datasets (WDBC, Parkinson's) are unchanged."
        ),
        rationale=(
            "Lets the classical L1 baseline support multiclass labels deterministically without "
            "altering its behaviour on binary datasets."
        ),
        references=("REQ-001",),
    ),
    FidelityNote(
        id="unified-run-rng-state",
        subject="Reinforced numbers come from the unified full-pipeline RNG state",
        gap=(
            "Runs are always full-pipeline (no standalone single-phase run is supported), so the "
            "shared RNG is already advanced by the earlier phases before the final comparison "
            "runs the reinforced methods."
        ),
        choice=(
            "The reported reinforced subsets/metrics are those produced from the post-earlier-phases "
            "(unified-run) RNG state. This is internally consistent: the interactive-feedback phase "
            "and the final comparison snapshot the same RNG state and agree."
        ),
        rationale=(
            "The unified run is the single source of truth for the headline; these numbers therefore "
            "differ from the standalone convergence-script numbers, which start from the initial "
            "post-build RNG state. Determinism is preserved (same seed reproduces the unified run)."
        ),
        references=("REQ-021", "AC-011", "CON-004"),
    ),
    FidelityNote(
        id="artifact-multi-file-layout",
        subject="Run artifact is split into per-dataset, per-seed files rather than one JSON",
        gap=("The specification calls for one self-contained JSON artifact per run."),
        choice=(
            "The artifact is written under experiments/<dataset>/seed-<n>/ as a selection file (the "
            "validation-surface results) and a separate test file (the held-out results). The layout "
            "reserves a per-seed dimension; PHASE-005 populates a single seed."
        ),
        rationale=(
            "Isolating the test results in their own file physically mirrors the gated, one-time test "
            "release (the selection surface never sees test), and the per-seed folder makes a future "
            "multi-seed run additive (new sibling folders, no migration)."
        ),
        references=("REQ-019-A", "AC-010", "RISK-002"),
    ),
)


def fidelity_notes_as_dicts() -> list[dict[str, Any]]:
    """Serialize the recorded register to a list of JSON-ready dicts (round-trip safe).

    Each note becomes ``{"id", "subject", "gap", "choice", "rationale", "references"}`` with
    ``references`` as a list of strings. All values are plain JSON types, so the block round-trips
    through ``json.dumps``/``json.loads`` unchanged — this is exactly what TASK-504 embeds as the
    artifact's ``fidelity_notes`` block (no reshaping required).
    """
    return [
        {
            "id": note.id,
            "subject": note.subject,
            "gap": note.gap,
            "choice": note.choice,
            "rationale": note.rationale,
            "references": list(note.references),
        }
        for note in FIDELITY_NOTES
    ]
