"""Documentation contracts for search strategies."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


STRATEGY_DIR = Path(__file__).parents[1] / "src" / "strategies"
REQUIRED_SECTIONS = ("Zweck:", "Mechanik:", "Grundlage:",
                     "Pipeline:", "Grenzen:")


@pytest.mark.parametrize("path", sorted(STRATEGY_DIR.glob("*.py")))
def test_strategy_module_and_class_docstrings_follow_the_contract(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_docstring = ast.get_docstring(tree)
    assert module_docstring and "\n" not in module_docstring
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not (node.name.endswith("Search") or node.name == "Pipeline"):
            continue
        docstring = ast.get_docstring(node)
        assert docstring is not None
        for section in REQUIRED_SECTIONS:
            assert section in docstring
