"""Capture the source actually present for a laboratory run."""

from __future__ import annotations

import hashlib
import platform
import shutil
from importlib.metadata import version
from pathlib import Path


def capture_source(output: Path) -> dict[str, object]:
    """Copy source beside a new report and record hashes and runtime versions.

    Existing reports or snapshots are never overwritten. Keep the source tree
    unchanged while a measurement runs; imported modules are not reloaded.
    """
    snapshot = output.parent / (output.stem + "-source")
    if output.exists() or snapshot.exists():
        raise FileExistsError(f"choose a fresh experiment output: {output}")
    root = Path(__file__).resolve().parents[1]
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("lab/**/*.py"),
        root / "run.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "source_directory": str(snapshot.resolve()),
        "source_sha256": hashes,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "numba", "matplotlib")},
    }
