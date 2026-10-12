"""Duplicate handler for the Phase-1 research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def duplicate_fingerprints(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    # hash_pandas_object hashes values and dtypes consistently within this
    # run.  It includes label/type and all original source columns, so a
    # target-only collision cannot silently be treated as a duplicate.
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype="uint64", copy=False)


class DuplicateRouter:
    """Keep duplicate rows of one source file in one final split.

    The source-file holdout strategy already keeps complete source files in a
    single fold.  This router is needed for IoT/Linux/Windows chronological
    blocks.  It is reset for every source file, which is intentional because
    source-file provenance is itself a split unit for the holdout platforms.
    """

    def __init__(self) -> None:
        self.first_split: dict[int, int] = {}
        self.first_seen = 0
        self.routed = 0
        self.cross_split = 0

    def route(self, fingerprints: np.ndarray, base_codes: np.ndarray) -> np.ndarray:
        final = base_codes.copy()
        for idx, digest in enumerate(fingerprints):
            key = int(digest)
            previous = self.first_split.get(key)
            if previous is None:
                self.first_split[key] = int(base_codes[idx])
                self.first_seen += 1
            else:
                if previous != int(base_codes[idx]):
                    self.cross_split += 1
                    self.routed += 1
                final[idx] = previous
        return final

    def route_masked(
        self,
        fingerprints: np.ndarray,
        base_codes: np.ndarray,
        eligible: np.ndarray,
        quarantine_code: int = -1,
    ) -> np.ndarray:
        """Route only eligible rows and leave quarantined rows untouched."""
        final = np.full(len(base_codes), quarantine_code, dtype=np.int8)
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.any():
            final[eligible] = self.route(fingerprints[eligible], base_codes[eligible])
        return final
