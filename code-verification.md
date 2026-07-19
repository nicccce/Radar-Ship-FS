# Code Verification - Git-Ready Alignment

Verified against the course **Git-Ready Code** rubric in `course-req/code.pdf`.
The rubric requires code that is understandable, runnable, testable, documented,
reproducible, and safe to share, with PEP 8 as the Python baseline.

## Current scorecard

| Rubric item | Status | Evidence |
|---|---|---|
| Clean checkout | Pass | The documented WDBC command completes from a fresh Git archive. |
| Folder structure | Pass | Domain code is separated under `src/`; tests live under `tests/`. |
| README | Pass | Purpose, install, data, run commands, outputs, checks, and license are documented. |
| Dependencies | Pass | Runtime dependencies are declared; the complete environment is pinned in `requirements.lock`. |
| Reproducibility | Pass | Seeded runs, recorded configuration, leakage-safe splits, and pinned installation are present. |
| Meaningful names | Pass | Public modules and APIs use domain-specific names. |
| Small units | Pass | Responsibilities are split by domain; obsolete warm-up orchestration was removed. |
| Documentation | Pass | Every source module and public class has a docstring. |
| Tests | Pass | `71 passed` on 2026-07-12. |
| PEP 8 enforcement | Pass | Ruff 0.15.21 is pinned; lint, format, import order, and line length are enforced. |
| Repository safety | Pass | Data, experiments, local environments, and course files are ignored; no secret pattern was found. |
| License | Pass | The repository contains an MIT license and exposes it in package metadata. |
| Commit communication | Pass going forward | The README requires short, specific, imperative commit subjects. |

## Verification commands

Run these from the repository root before committing:

```bash
ruff format --check .
ruff check .
python -m compileall -q src tests
pytest
python src/run_irfs.py --dataset wdbc --seeds 42
```

The reproducible installation path is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

## Verified details

- Ruff checks all Python files at a declared maximum line length of 110 characters.
- `ruff check .`, `ruff format --check .`, `compileall`, and `pytest` pass.
- `data/`, `experiments/`, and `course-req/` are ignored and untracked.
- No `TODO`, `FIXME`, or `XXX` markers remain under `src/`.
- Runtime progress messages are intentional CLI output, not debugging prints.
- Same-seed reproducibility is tested in `tests/test_invariants.py`; runtime output
  points to that existing file.
- Historical vague commit subjects remain unchanged to avoid rewriting shared Git
  history. New commits follow the specific-message rule documented in the README.

## Scope of this verification

This document evaluates Git readiness and code quality. It does not claim that a
single experimental seed establishes scientific superiority; performance conclusions
must be based on the configured multi-seed aggregate.
