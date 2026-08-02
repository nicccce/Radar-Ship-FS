"""CLI for configuration-driven stable experiments."""

from __future__ import annotations

import argparse
import json

from radar_ship_fs.experiment.config import load_experiment_spec
from radar_ship_fs.experiment.runner import ExperimentRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m radar_ship_fs.experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "dry-run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--seed", action="append", type=int, default=[])
        child.add_argument("--method", action="append", default=[])
        if command == "run":
            resume = child.add_mutually_exclusive_group()
            resume.add_argument("--resume", action="store_true", dest="resume")
            resume.add_argument("--no-resume", action="store_false", dest="resume")
            child.set_defaults(resume=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = load_experiment_spec(args.config)
    runner = ExperimentRunner(spec, seed_filter=args.seed, method_filter=args.method)
    if args.command == "dry-run":
        print(json.dumps(runner.dry_run(), ensure_ascii=False, indent=2))
        return
    runner.run(resume=args.resume)


if __name__ == "__main__":
    main()
