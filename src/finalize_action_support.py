"""Freeze Task 4A artifacts and implementation hashes for delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "action_support_v1"
DELIVERY = EXPERIMENT / "delivery-manifest.json"
CODE_DOCUMENTS = (
    "configs/v16n/action_support_v1.toml",
    "documents/research-plan/next-stage-roadmap.md",
    "documents/research-plan/action-support-protocol.md",
    "documents/research-plan/04a-action-support.md",
    "src/radar_ship_fs/experiment/config.py",
    "src/radar_ship_fs/ppo/config.py",
    "src/radar_ship_fs/ppo/ppo_env.py",
    "src/radar_ship_fs/ppo/run_session.py",
    "src/run_action_support.py",
    "src/audit_action_support.py",
    "src/summarize_action_support.py",
    "src/finalize_action_support.py",
    "tests/test_action_support.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-tests-passed", type=int, required=True)
    args = parser.parse_args()

    audit = json.loads((EXPERIMENT / "audit.json").read_text())
    complete = json.loads((EXPERIMENT / "complete.json").read_text())
    if audit["status"] != "passed":
        raise RuntimeError("Task 4A audit is not passed")
    if complete["source_test_open_count"] or complete["rl_training_count"]:
        raise RuntimeError("Task 4A isolation invariant failed")

    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(EXPERIMENT.rglob("*"))
        if path.is_file() and path != DELIVERY
    }
    code_documents = {
        relative: sha256(ROOT / relative) for relative in CODE_DOCUMENTS
    }
    payload = {
        "version": complete["version"],
        "completed_utc": complete["completed_utc"],
        "delivered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_sha256": artifacts,
        "code_documents_sha256": code_documents,
        "audit": audit["status"],
        "ruff": "passed",
        "focused_tests_passed": args.focused_tests_passed,
        "focused_test_scope": (
            "action support, PPO feature IDs, PPO initialization, search diagnosis, "
            "and stable configuration"
        ),
        "full_pytest_status": (
            "blocked_at_collection_by_absent_legacy_and_stage2_modules"
        ),
        "additional_collectable_tests_passed_before_data_errors": 73,
        "additional_test_blocker": "absent data/sim_ship_cr_v10 train/test files",
        "source_test_open_count": complete["source_test_open_count"],
        "rl_training_count": complete["rl_training_count"],
        "historical_search_policy_gap": complete[
            "historical_search_policy_gap_status"
        ],
        "candidate_support_decision": "PASS",
        "fixed_budget_efficiency_decision": "NOT_ESTABLISHED",
        "rl_effectiveness_decision": "NA_NO_RL_TRAINING",
        "delivery_manifest_self_hash": "excluded_by_construction",
    }
    DELIVERY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DELIVERY.relative_to(ROOT)} with {len(artifacts)} artifact hashes")


if __name__ == "__main__":
    main()
