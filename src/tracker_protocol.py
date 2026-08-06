"""Protocol interface for energy trackers.

GramTracker and AutocorrTracker both implement this implicitly.
The solver uses duck-typing — a ``Protocol`` documents the contract
but isn't enforced at runtime.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Tracker(Protocol):
    """Duck-type contract: any tracker with these attributes works with solver."""

    _rows_band: Any
    _cols_band: Any

    def build(
        self,
        seqs: npt.NDArray[np.int8],
        band_rows: Any = None,
        band_cols: Any = None,
    ) -> None: ...

    def energy(self) -> int: ...

    def flip(self, _seqs: npt.NDArray[np.int8], s: int, c: int) -> int: ...

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int: ...

    def _combo_delta(
        self, bands: list[tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]]
    ) -> int: ...
