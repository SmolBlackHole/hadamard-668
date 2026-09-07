# Writing and maintaining documentation

Parent: [Documentation index](README.md)

## Contents

- [Writing and maintaining documentation](#writing-and-maintaining-documentation)
  - [Contents](#contents)
  - [Choose the owner](#choose-the-owner)
  - [Keep navigation useful](#keep-navigation-useful)
  - [Describe evidence precisely](#describe-evidence-precisely)
  - [Document code contracts](#document-code-contracts)

## Choose the owner

Write in English. Give each fact one authoritative page and link to it instead of
copying paragraphs. The root README owns the purpose and quick start, the documentation
index owns navigation, and each topic page owns the question listed in the
[reading guide](README.md#reading-guide).

Keep implementation descriptions aligned with code and tests. Do not add empty
starter pages or speculative architecture to satisfy a template.

## Keep navigation useful

Every documentation page starts with a `Parent:` link to its nearest index.
Pages with several sections include a linked table of contents. Link new pages
from the index and update incoming links when renaming headings or moving files.

Keep the README short enough for a new contributor to find a working first command.
Explain symbols, matrix order versus sequence length, and assumptions where needed.

## Describe evidence precisely

Separate mathematical identities, tested implementations, numerical observations
and hypotheses. Preserve formulas, normalization conventions and their domains
when editing prose. Do not turn a finite numerical check into a proof.

Record known limitations beside the affected behavior. Describe future work as
future work, not an implemented capability. Experiment counts and performance
claims require retained artifacts and a reproducible command.

## Document code contracts

Python docstrings use English and the Google convention. Start with a concise
contract. Add `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Note:` or `Warning:`
only when they carry useful information. Type annotations own types; docstrings
explain shape, units, ordering, mutation, ownership and domain meaning.

Public modules, classes and functions need docstrings. Private helpers need them
when the contract is not clear from the signature and implementation. Tests are
exempt because their names describe the behavior under test. Ruff checks this
boundary in the [quality workflow](development.md#quality-checks).
