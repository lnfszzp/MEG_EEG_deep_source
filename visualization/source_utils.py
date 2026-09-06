from __future__ import annotations

from pathlib import Path

import numpy as np


def load_masked_source(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=True)
    source = np.asarray(result["S"], dtype=float)
    mask = np.asarray(result["region_mask"], dtype=bool).ravel()
    return source * mask[:, None]
