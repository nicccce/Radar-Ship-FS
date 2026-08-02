"""Dependency-boundary checks for the namespaced stable implementation."""

from __future__ import annotations

import ast
from pathlib import Path

from config import load_config
from harness.contract import Selector
from radar_ship_fs.legacy import build_legacy_selector


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_stable_package_never_imports_legacy_or_stage2_scripts() -> None:
    root = Path("src/radar_ship_fs")
    violations: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        if "legacy" in path.parts:
            continue
        forbidden = sorted(
            name
            for name in _imports(path)
            if name == "radar_ship_fs.legacy"
            or name.startswith("radar_ship_fs.legacy.")
            or name.startswith("run_stage2_")
        )
        if forbidden:
            violations[str(path)] = forbidden
    assert violations == {}


def test_frozen_legacy_engine_is_exposed_only_through_selector_contract() -> None:
    selector = build_legacy_selector("marlfs", load_config({"seeds": (42,)}))
    assert isinstance(selector, Selector)


def test_all_stage2_compatibility_entries_are_thin_wrappers() -> None:
    entries = sorted(Path("src").glob("run_stage2_*.py"))
    assert entries
    for path in entries:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert definitions == [], path
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 15
        assert any(name.startswith("radar_ship_fs.legacy.stage2") for name in _imports(path))
