#!/usr/bin/env python3
"""Strict, dependency-free validator for the inert H668 run manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from .verify import InvalidMatrix, read_regular_file


MAX_MANIFEST_BYTES = 32 * 1024
MAX_U64 = 2**64 - 1
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
DECIMAL = re.compile(r"^(0|[1-9][0-9]*)$")

REQUIRED_KEYS = {
    "schema_version",
    "result_type",
    "method_family",
    "search_scope",
    "coverage_kind",
    "seed_derivation",
    "seeds",
    "evaluations",
    "wall_seconds",
    "hardware_summary",
    "model_summary",
    "code_url",
    "code_commit",
    "parent_candidate_sha256",
    "candidate_sha256",
    "metrics",
    "publication_consent",
}
METHOD_FAMILIES = {
    "gs_sds",
    "williamson_propus",
    "sat_cp_pb",
    "alternative_algebraic",
    "local_search",
    "evolutionary_gpu",
    "literature_transfer",
    "hybrid",
    "other",
}
METRIC_KEYS = {
    "off_diagonal_energy",
    "orthogonal_row_pairs",
    "max_absolute_off_diagonal",
}


class InvalidManifest(ValueError):
    """Raised when a manifest violates the published acceptance schema."""


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidManifest(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> object:
    raise InvalidManifest(f"non-standard JSON number is forbidden: {value}")


def _bounded_string(value: object, field: str, limit: int, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str):
        raise InvalidManifest(
            f"{field} must be a string" + (" or null" if nullable else ""))
    if len(value) > limit:
        raise InvalidManifest(f"{field} exceeds {limit} characters")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
        raise InvalidManifest(
            f"{field} contains a forbidden control character")


def _nullable_nonnegative_number(value: object, field: str, integer: bool = False) -> None:
    if value is None:
        return
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise InvalidManifest(
            f"{field} must be a non-negative {'integer' if integer else 'number'} or null")
    if (not integer and not math.isfinite(value)) or value < 0:
        raise InvalidManifest(f"{field} must be non-negative")


def load_and_validate_manifest(path: Path) -> dict[str, object]:
    try:
        raw = read_regular_file(path)
    except InvalidMatrix as exc:
        raise InvalidManifest(str(exc)) from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InvalidManifest(f"run.json exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidManifest("run.json must be valid UTF-8") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonstandard_number,
        )
    except (json.JSONDecodeError, InvalidManifest, RecursionError, ValueError) as exc:
        raise InvalidManifest(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidManifest("manifest root must be a JSON object")
    keys = set(data)
    if keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        raise InvalidManifest(
            f"manifest keys do not match schema; missing={missing}, extra={extra}")

    if data["schema_version"] != "h668-run-v1":
        raise InvalidManifest("schema_version must be 'h668-run-v1'")
    if data["result_type"] not in {"exact_solution", "checkpoint"}:
        raise InvalidManifest("invalid result_type")
    if data["method_family"] not in METHOD_FAMILIES:
        raise InvalidManifest("invalid method_family")
    if data["coverage_kind"] not in {"heuristic", "exhaustive_claim", "certified"}:
        raise InvalidManifest("invalid coverage_kind")

    _bounded_string(data["search_scope"], "search_scope", 2000)
    _bounded_string(data["seed_derivation"],
                    "seed_derivation", 512, nullable=True)
    _bounded_string(data["hardware_summary"],
                    "hardware_summary", 256, nullable=True)
    _bounded_string(data["model_summary"], "model_summary", 256, nullable=True)

    seeds = data["seeds"]
    if not isinstance(seeds, list) or len(seeds) > 1024:
        raise InvalidManifest(
            "seeds must be an array with at most 1024 entries")
    for index, seed in enumerate(seeds):
        if not isinstance(seed, str) or not DECIMAL.fullmatch(seed):
            raise InvalidManifest(
                f"seeds[{index}] must be a canonical unsigned decimal string")
        if int(seed) > MAX_U64:
            raise InvalidManifest(
                f"seeds[{index}] exceeds unsigned 64-bit range")

    _nullable_nonnegative_number(
        data["evaluations"], "evaluations", integer=True)
    _nullable_nonnegative_number(data["wall_seconds"], "wall_seconds")

    code_url = data["code_url"]
    _bounded_string(code_url, "code_url", 2048, nullable=True)
    if code_url is not None:
        parsed = urlparse(code_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise InvalidManifest(
                "code_url must be an absolute HTTPS URL or null")
    code_commit = data["code_commit"]
    if code_commit is not None and (
        not isinstance(code_commit, str) or not COMMIT.fullmatch(code_commit)
    ):
        raise InvalidManifest(
            "code_commit must be 7-64 hexadecimal characters or null")
    candidate_hash = data["candidate_sha256"]
    if not isinstance(candidate_hash, str) or not HEX_64.fullmatch(candidate_hash):
        raise InvalidManifest(
            "candidate_sha256 must be a lowercase SHA-256 hex digest")
    parent_hash = data["parent_candidate_sha256"]
    if parent_hash is not None and (
        not isinstance(parent_hash, str) or not HEX_64.fullmatch(parent_hash)
    ):
        raise InvalidManifest(
            "parent_candidate_sha256 must be a lowercase SHA-256 hex digest or null"
        )
    if data["publication_consent"] is not True:
        raise InvalidManifest("publication_consent must be true")

    metrics = data["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise InvalidManifest("metrics keys do not match the schema")
    for key in METRIC_KEYS:
        _nullable_nonnegative_number(
            metrics[key], f"metrics.{key}", integer=True)
    if metrics["orthogonal_row_pairs"] is not None and metrics["orthogonal_row_pairs"] > 222_778:
        raise InvalidManifest("claimed orthogonal_row_pairs exceeds 222778")
    if (
        metrics["max_absolute_off_diagonal"] is not None
        and metrics["max_absolute_off_diagonal"] > 668
    ):
        raise InvalidManifest("claimed max_absolute_off_diagonal exceeds 668")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an H668 run.json manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-filename", action="store_true")
    args = parser.parse_args()
    try:
        if args.require_filename and args.manifest.name != "run.json":
            raise InvalidManifest(
                "manifest filename must be exactly 'run.json'")
        data = load_and_validate_manifest(args.manifest)
    except InvalidManifest as exc:
        print(json.dumps(
            {"valid_manifest": False, "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "valid_manifest": True,
                "schema_version": data["schema_version"],
                "result_type": data["result_type"],
                "candidate_sha256": data["candidate_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
