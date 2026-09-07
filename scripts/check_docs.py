"""Lint maintained Markdown and check its math for known GitHub rendering hazards.

PyMarkdown owns Markdown linting; markdown-it parses fences and inline math;
latex2mathml checks conversion. The small compatibility rules cover observed
GitHub failures that a standalone TeX renderer accepts. This is not a complete
emulation of GitHub's changing sanitizer and MathJax configuration.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from latex2mathml.converter import convert  # pyright: ignore[reportUnknownVariableType]
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parents[1]


def check_math(expression: str) -> list[str]:
    r"""Check one TeX expression, without Markdown or dollar delimiters.

    Literal comparisons must use ``\lt`` and ``\gt`` because GitHub can treat
    raw angle brackets as HTML, including silently removing parts of cases.
    """
    errors: list[str] = []
    if "<" in expression or ">" in expression:
        errors.append(r"raw angle bracket in math; use \lt or \gt to avoid GitHub HTML parsing")
    if re.search(r"\\operatorname\b", expression):
        errors.append(r"GitHub rejects \operatorname here; use \mathrm for these operator labels")
    depth = 0
    for token in re.findall(r"\\(?:[a-zA-Z]+|.)|[{}]", expression):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                break
    if depth:
        errors.append("unbalanced TeX braces")
    if errors:
        return errors
    try:
        mathml = ElementTree.fromstring(cast(str, convert(expression)))
    except Exception as error:  # Converter exposes several unrelated exception types.
        return [f"TeX conversion failed: {type(error).__name__}: {error}"]
    unknown = sorted(set(re.findall(r"\\[a-zA-Z]+", "".join(mathml.itertext()))))
    if unknown:
        errors.append(f"unconverted TeX commands: {', '.join(unknown)}")
    return errors


def check_document(source: str) -> list[tuple[int, str]]:
    """Return line-numbered math diagnostics, ignoring ordinary code fences."""
    parser = MarkdownIt("commonmark").use(dollarmath_plugin)
    errors: list[tuple[int, str]] = []
    lines = source.splitlines()
    for token in parser.parse(source):
        line = token.map[0] + 1 if token.map else 1
        if token.type == "fence" and token.info.strip() == "math":
            if token.map and lines[token.map[1] - 1].strip() != token.markup:
                errors.append((line, "unclosed math fence"))
            errors.extend((line + 1, error) for error in check_math(token.content))
        elif token.type.startswith("math_block"):
            errors.append((line, "use a fenced math block instead of $$ display delimiters"))
        elif token.type == "inline":
            for child in token.children or []:
                if child.type == "math_inline":
                    errors.extend((line, error) for error in check_math(child.content))
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run package-backed Markdown lint and math checks on maintained files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Explicit Markdown files to check.")
    args = parser.parse_args(argv)
    files: list[Path] = args.paths or sorted([*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")])
    files = [path.resolve() for path in files]
    lint = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymarkdown",
            "--config",
            str(ROOT / ".pymarkdown.json"),
            "scan",
            *(str(path) for path in files),
        ],
        cwd=ROOT,
        check=False,
    )
    failures = 0
    for path in files:
        for line, message in check_document(path.read_text(encoding="utf-8")):
            print(f"{path}:{line}: math: {message}")
            failures += 1
    if not lint.returncode and not failures:
        print(f"Documentation checks passed ({len(files)} files).")
    return 1 if lint.returncode or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
