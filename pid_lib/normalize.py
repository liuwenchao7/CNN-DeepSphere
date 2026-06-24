"""Per-channel normalization estimated on training data only."""


import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from pid_lib.config import FEATURE_NAMES
from pid_lib.data_io import FileEntry, load_aligned_file, mark_no_hit_features


def estimate_norm_stats(
    entries,
    max_events: Optional[int] = None,
) -> Dict:
    """Compute per-feature mean/std ignoring NaN sentinels."""
    sums = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    sq_sums = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    counts = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    seen = 0

    for entry in entries:
        x, _, _ = load_aligned_file(entry)
        x, _ = mark_no_hit_features(x)
        for c in range(len(FEATURE_NAMES)):
            col = x[..., c].ravel()
            valid = col[~np.isnan(col)]
            if valid.size:
                sums[c] += valid.sum()
                sq_sums[c] += (valid ** 2).sum()
                counts[c] += valid.size
        seen += x.shape[0]
        if max_events is not None and seen >= max_events:
            break

    mean = np.where(counts > 0, sums / counts, 0.0)
    var = np.where(counts > 0, sq_sums / counts - mean ** 2, 1.0)
    std = np.sqrt(np.maximum(var, 1e-8))

    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "counts": counts.tolist(),
        "method": "per_channel_zscore_ignore_nan",
    }
    return stats


def save_norm_stats(output_dir: str, stats: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "norm_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path


def load_norm_stats(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_norm(x: np.ndarray, stats: Dict, fill_nan: float = 0.0) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    out = x.astype(np.float32, copy=True)
    for c in range(len(FEATURE_NAMES)):
        col = out[..., c]
        bad = np.isnan(col)
        col = (col - mean[c]) / std[c]
        col[bad] = fill_nan
        out[..., c] = col
    return out
