from pathlib import Path

import pytest

from scripts.check_docs import check_document, check_math, main


@pytest.mark.parametrize(
    "expression",
    [r"\sum_{i<j} x_i", r"\begin{cases}x,&h<n-h\\-x,&h>n-h\end{cases}"],
)
def test_github_html_hazards_are_rejected(expression: str) -> None:
    assert any("raw angle bracket" in error for error in check_math(expression))


@pytest.mark.parametrize(
    "expression",
    [
        r"\sum_{i\lt j}\langle H_i,H_j\rangle^2",
        r"\begin{cases}x,&h\lt n-h\\-x,&h\gt n-h\end{cases}",
        r"x\in\{-1,+1\}^{n}",
    ],
)
def test_github_safe_math_passes(expression: str) -> None:
    assert check_math(expression) == []


@pytest.mark.parametrize("expression", [r"\frac{1}{2", r"x}", r"\unknowncommand{x}"])
def test_malformed_tex_is_rejected(expression: str) -> None:
    assert check_math(expression)


def test_math_is_checked_but_code_examples_are_ignored() -> None:
    source = "```python\nx = 'i<j'\n```\n\nInline $i<j$.\n\n```math\nx}\n```\n"
    errors = check_document(source)
    assert [line for line, _ in errors] == [5, 8]


def test_display_delimiters_and_unclosed_fence_are_rejected() -> None:
    assert any("fenced math" in error for _, error in check_document("$$x$$\n"))
    assert any("unclosed" in error for _, error in check_document("```math\nx\n"))


def test_cli_runs_markdown_and_math_checks(tmp_path: Path) -> None:
    document = tmp_path / "sample.md"
    document.write_text("# Sample\n\n```math\nx^2\n```\n", encoding="utf-8")
    assert main([str(document)]) == 0
    document.write_text("# Sample\n\n```math\ni<j\n```\n", encoding="utf-8")
    assert main([str(document)]) == 1
    document.write_text("# Sample\n\n### Skipped heading\n", encoding="utf-8")
    assert main([str(document)]) == 1
