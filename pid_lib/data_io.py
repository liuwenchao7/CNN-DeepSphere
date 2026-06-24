"""Read-only helpers for discovering and loading PID training data."""


import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pid_lib.config import (
    CLASS_LABEL,
    CLASS_ROOTS,
    FEATURE_FILE_PREFIX,
    FEATURE_NAMES,
    FHT_NO_HIT_MAX,
    N_PMT,
    PEAKTIME_NO_HIT_MIN,
    SUBDIRS,
)

_FILE_ID_RE = re.compile(r"_(\d+)\.(npy|npz)$")


@dataclass
class FileEntry:
    cls_name: str
    file_id: int
    label: int
    root: str

    @property
    def elec_dir(self) -> str:
        return os.path.join(self.root, "elec_fea")

    @property
    def y_path(self) -> str:
        return os.path.join(self.root, "y", f"y_{self.file_id}.npy")

    def feature_path(self, feat: str) -> str:
        return os.path.join(self.elec_dir, f"{FEATURE_FILE_PREFIX[feat]}_{self.file_id}.npy")


def parse_file_id(filename: str) -> Optional[int]:
    m = _FILE_ID_RE.search(filename)
    return int(m.group(1)) if m else None


def list_subdir_files(root: str, subdir: str, pattern: str = "*") -> List[str]:
    path = os.path.join(root, subdir)
    if not os.path.isdir(path):
        return []
    import glob

    return sorted(glob.glob(os.path.join(path, pattern)))


def discover_file_ids(root: str, subdir: str, prefix: str) -> List[int]:
    ids = []
    for f in list_subdir_files(root, subdir, f"{prefix}_*"):
        fid = parse_file_id(os.path.basename(f))
        if fid is not None:
            ids.append(fid)
    return sorted(set(ids))


def required_feature_paths(entry: FileEntry) -> Dict[str, str]:
    paths = {feat: entry.feature_path(feat) for feat in FEATURE_NAMES}
    paths["y"] = entry.y_path
    return paths


def is_file_complete(entry: FileEntry) -> Tuple[bool, List[str]]:
    missing = []
    for feat, p in required_feature_paths(entry).items():
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            missing.append(feat)
    return len(missing) == 0, missing


def is_file_loadable(entry: FileEntry) -> bool:
    if not is_file_complete(entry)[0]:
        return False
    try:
        load_aligned_file(entry)
        return True
    except Exception:
        return False


def discover_valid_entries(cls_name: Optional[str] = None) -> List[FileEntry]:
    entries: List[FileEntry] = []
    classes = [cls_name] if cls_name else list(CLASS_ROOTS.keys())
    for name in classes:
        root = CLASS_ROOTS[name]
        label = CLASS_LABEL[name]
        # Intersect file ids that have all six features and y.
        id_sets = []
        for feat in FEATURE_NAMES:
            prefix = FEATURE_FILE_PREFIX[feat]
            id_sets.append(set(discover_file_ids(root, "elec_fea", prefix)))
        id_sets.append(set(discover_file_ids(root, "y", "y")))
        common = set.intersection(*id_sets) if id_sets else set()
        for fid in sorted(common):
            entry = FileEntry(cls_name=name, file_id=fid, label=label, root=root)
            ok, _ = is_file_complete(entry)
            if ok:
                entries.append(entry)
    return entries


def _safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path, allow_pickle=True)
    except (OSError, ValueError, EOFError):
        return None


def load_y_arrays(entry: FileEntry) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y = _safe_load_npy(entry.y_path)
    if y is None:
        raise ValueError(f"Cannot load y file: {entry.y_path}")
    y_pmt_path = os.path.join(entry.elec_dir, f"y_pmt_{entry.file_id}.npy")
    y_pmt = _safe_load_npy(y_pmt_path) if os.path.isfile(y_pmt_path) else None
    return y, y_pmt


def align_event_indices(y: np.ndarray, y_pmt: Optional[np.ndarray]) -> np.ndarray:
    """Return row indices into y (and elec_fea rows) to keep."""
    n = y.shape[0]
    if y_pmt is None or y_pmt.shape[0] == n:
        return np.arange(n, dtype=np.int64)
    # Match on shared label columns: evt(1), theta(2), phi(3), pid(4), energy(5).
    keys_y = y[:, 1:6]
    keys_pmt = y_pmt[:, 1:6]
    keep = []
    for i, key in enumerate(keys_pmt):
        match = np.where(np.all(np.isclose(keys_y, key, rtol=0, atol=0), axis=1))[0]
        if len(match) == 1:
            keep.append(int(match[0]))
    return np.array(keep, dtype=np.int64)


def load_features_for_file(
    entry: FileEntry,
    row_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load stacked features (n_event, N_PMT, 6) and labels (n_event,)."""
    arrays = []
    for feat in FEATURE_NAMES:
        arr = _safe_load_npy(entry.feature_path(feat))
        if arr is None:
            raise ValueError(f"Cannot load feature {feat} for {entry.cls_name} {entry.file_id}")
        if row_indices is not None:
            arr = arr[row_indices]
        arrays.append(arr)
    x = np.stack(arrays, axis=-1).astype(np.float32)
    labels = np.full(x.shape[0], entry.label, dtype=np.int32)
    return x, labels


def load_aligned_file(entry: FileEntry) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load one file with y/y_pmt alignment when counts differ."""
    y, y_pmt = load_y_arrays(entry)
    meta = {
        "n_y": int(y.shape[0]),
        "n_y_pmt": int(y_pmt.shape[0]) if y_pmt is not None else None,
        "dropped": 0,
        "drop_reason": None,
    }
    if y_pmt is not None and y_pmt.shape[0] != y.shape[0]:
        idx = align_event_indices(y, y_pmt)
        meta["aligned_to"] = "y_pmt"
        meta["n_kept"] = int(len(idx))
        meta["dropped"] = int(y_pmt.shape[0] - len(idx))
        if len(idx) == 0:
            raise ValueError(f"No aligned events for {entry.cls_name} file {entry.file_id}")
        x, labels = load_features_for_file(entry, row_indices=idx)
        meta["drop_reason"] = "y/y_pmt count mismatch"
        return x, labels, meta
    x, labels = load_features_for_file(entry)
    meta["n_kept"] = int(x.shape[0])
    return x, labels, meta


def mark_no_hit_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace FHT/peaktime sentinels with NaN; return mask (n_event, n_feat)."""
    mask = np.ones(x.shape[:2] + (len(FEATURE_NAMES),), dtype=bool)
    fht_idx = FEATURE_NAMES.index("fht")
    pt_idx = FEATURE_NAMES.index("peaktime")
    fht_bad = x[..., fht_idx] <= FHT_NO_HIT_MAX
    pt_bad = x[..., pt_idx] >= PEAKTIME_NO_HIT_MIN
    x = x.copy()
    x[..., fht_idx][fht_bad] = np.nan
    x[..., pt_idx][pt_bad] = np.nan
    mask[..., fht_idx][fht_bad] = False
    mask[..., pt_idx][pt_bad] = False
    return x, mask


def summarize_npz(path: str) -> Dict:
    data = np.load(path, allow_pickle=False)
    info = {"path": path, "keys": list(data.keys())}
    for k in data.keys():
        arr = data[k]
        info[k] = {"shape": arr.shape, "dtype": str(arr.dtype)}
    return info


def inspect_npy(path: str, max_rows: int = 2) -> Dict:
    if not path.endswith(".npy"):
        return {"path": path, "error": "not npy"}
    try:
        arr = np.load(path, mmap_mode="r")
    except ValueError:
        arr = np.load(path, allow_pickle=True)
    info = {
        "path": path,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }
    if arr.dtype == object:
        info["object_element"] = str(type(arr.flat[0])) if arr.size else "empty"
        return info
    if np.issubdtype(arr.dtype, np.floating):
        info["has_nan"] = bool(np.isnan(arr[:max_rows]).any())
        info["has_inf"] = bool(np.isinf(arr[:max_rows]).any())
    return info


def reconstruct_event_from_compact_npz(
    npz_path: str,
    event_idx: int,
    n_pmt: int = N_PMT,
) -> Tuple[np.ndarray, Dict]:
    """Rebuild dense (n_pmt, n_samples) waveform for one event from compact NPZ."""
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.keys())
    meta = {"keys": list(keys), "event_idx": event_idx}

    if "waveform" in keys and "event_offsets" in keys and "copyNo" in keys:
        if "waveform_offsets" in keys:
            # Ragged per-PMT variable length (decon_waveform style).
            evt_off = int(data["event_offsets"][event_idx])
            evt_end = int(data["event_offsets"][event_idx + 1])
            copy = data["copyNo"][evt_off:evt_end].astype(np.int64)
            wf_off = data["waveform_offsets"][evt_off : evt_end + 1]
            wf_flat = data["waveform"]
            max_len = 0
            segments = []
            for i in range(len(copy)):
                s = int(wf_off[i])
                e = int(wf_off[i + 1])
                if e > s + 1:
                    seg = wf_flat[s + 1 : e].astype(np.float32)
                    segments.append((int(copy[i]), seg))
                    max_len = max(max_len, len(seg))
            dense = np.zeros((n_pmt, max_len), dtype=np.float32)
            for pmt, seg in segments:
                dense[pmt, : len(seg)] = seg
            meta["format"] = "compact_ragged"
            meta["n_hit_pmt"] = len(copy)
            meta["n_samples"] = max_len
            meta["variable_pmt_length"] = True
            return dense, meta

        # Fixed-length per hit (waveform_*.npz style).
        evt_off = int(data["event_offsets"][event_idx])
        evt_end = int(data["event_offsets"][event_idx + 1])
        copy = data["copyNo"][evt_off:evt_end].astype(np.int64)
        wf = data["waveform"][evt_off:evt_end]
        n_samples = wf.shape[1] if wf.ndim == 2 else int(data.get("n_samples", 1008))
        dense = np.zeros((n_pmt, n_samples), dtype=np.float32)
        dense[copy] = wf.astype(np.float32)
        meta["format"] = "compact_fixed"
        meta["n_hit_pmt"] = len(copy)
        meta["n_samples"] = n_samples
        return dense, meta

    raise ValueError(f"Unsupported NPZ layout: {npz_path} keys={list(keys)}")


def load_det_fea_event(npz_path: str, event_idx: int) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    evt_off = int(data["event_offsets"][event_idx])
    evt_end = int(data["event_offsets"][event_idx + 1])
    pmt_off = data["pmt_offsets"][evt_off:evt_end]
    pmt_end = data["pmt_offsets"][evt_off + 1 : evt_end + 1]
    times, npes = [], []
    for s, e in zip(pmt_off, pmt_end):
        times.append(data["time"][s:e])
        npes.append(data["npe"][s:e])
    return {
        "pmtID": data["pmtID"][evt_off:evt_end],
        "time": times,
        "npe": npes,
        "simEventID": int(data["simEventID"][event_idx]),
    }


def find_waveform_files(root: str, file_id: int) -> Dict[str, Optional[str]]:
    wf_dir = os.path.join(root, "waveform")
    candidates = {
        "waveform": [
            f"waveform_{file_id}.npz",
            f"waveform_{file_id}_compact.npz",
        ],
        "decon_waveform": [f"decon_waveform_{file_id}.npz"],
        "decon_npevst": [f"decon_npevst_{file_id}.npy", f"decon_npevst_{file_id}.npz"],
    }
    out = {}
    for kind, names in candidates.items():
        out[kind] = None
        for name in names:
            p = os.path.join(wf_dir, name)
            if os.path.isfile(p):
                out[kind] = p
                break
    return out
