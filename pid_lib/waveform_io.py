"""Load per-event PMT waveforms for CNN+DeepSphere PID."""


import os
from typing import Dict, Optional, Tuple

import numpy as np

from pid_lib.config import N_PMT, WFSAMPLING_OUTPUT_ROOT
from pid_lib.data_io import FileEntry, find_waveform_files, reconstruct_event_from_compact_npz


def load_decon_waveform(entry: FileEntry, event_idx: int, max_len: int) -> np.ndarray:
    """Return (N_PMT, max_len, 1) float32."""
    paths = find_waveform_files(entry.root, entry.file_id)
    path = paths.get("decon_waveform")
    if not path:
        raise FileNotFoundError(f"decon_waveform missing for {entry.cls_name} {entry.file_id}")
    dense, meta = reconstruct_event_from_compact_npz(path, event_idx)
    out = np.zeros((N_PMT, max_len, 1), dtype=np.float32)
    n = min(max_len, dense.shape[1])
    out[:, :n, 0] = dense[:, :n]
    return out


def load_waveform(entry: FileEntry, event_idx: int, max_len: int) -> np.ndarray:
    """Return raw waveform as (N_PMT, max_len, 1) float32."""
    paths = find_waveform_files(entry.root, entry.file_id)
    path = paths.get("waveform")
    if not path:
        raise FileNotFoundError(f"waveform missing for {entry.cls_name} {entry.file_id}")
    dense, _ = reconstruct_event_from_compact_npz(path, event_idx)
    dense = 16384.0 - dense
    out = np.zeros((N_PMT, max_len, 1), dtype=np.float32)
    n = min(max_len, dense.shape[1])
    out[:, :n, 0] = dense[:, :n]
    return out


def parse_decon_npevst_hits(
    hits,
    max_points: int,
    t_max: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse one event's decon_npevst hits into (N_PMT, max_points, 2)."""
    out = np.zeros((N_PMT, max_points, 2), dtype=np.float32)
    lengths = np.zeros(N_PMT, dtype=np.int32)
    if hits is None or len(hits) == 0:
        return out, lengths
    # columns: npe, time, pmt_id
    npe = hits[:, 0].astype(np.float32)
    time = hits[:, 1].astype(np.float32)
    pmt = hits[:, 2].astype(np.int64)
    valid = (pmt >= 0) & (pmt < N_PMT) & (time >= 0) & (time <= t_max)
    npe, time, pmt = npe[valid], time[valid], pmt[valid]
    if len(pmt) == 0:
        return out, lengths

    for pid in np.unique(pmt):
        idx = np.where(pmt == pid)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(time[idx])]
        n = min(len(order), max_points)
        lengths[pid] = n
        tt = np.clip(time[order[:n]] / max(t_max, 1.0), 0.0, 1.0)
        qq = npe[order[:n]]
        qmax = max(float(np.max(np.abs(qq))), 1.0)
        out[pid, :n, 0] = tt
        out[pid, :n, 1] = qq / qmax
    return out, lengths


def decon_npevst_event_from_file(
    file_arr,
    event_idx: int,
    max_points: int,
    t_max: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one event from a preloaded decon_npevst file array."""
    return parse_decon_npevst_hits(file_arr[event_idx], max_points, t_max=t_max)


def load_decon_npevst(entry: FileEntry, event_idx: int, max_points: int, t_max: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load sparse decon_npevst points into (N_PMT, max_points, 2).

    Channels:
      - [:, :, 0] : normalized time in [0, 1]
      - [:, :, 1] : normalized nPE (per PMT max)
    """
    paths = find_waveform_files(entry.root, entry.file_id)
    path = paths.get("decon_npevst")
    if not path:
        raise FileNotFoundError(f"decon_npevst missing for {entry.cls_name} {entry.file_id}")
    file_arr = np.load(path, allow_pickle=True)
    return decon_npevst_event_from_file(file_arr, event_idx, max_points, t_max=t_max)


def wfsampling_event_from_file(
    cached,
    event_idx: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one event from a preloaded WFSampling file cache."""
    fmt, data = cached
    wave = np.zeros((N_PMT, max_points, 2), dtype=np.float32)
    lengths = np.zeros(N_PMT, dtype=np.int32)

    if fmt == "npy":
        t_evt = data["time"][event_idx]
        a_evt = data["adc"][event_idx]
        o_evt = data["offsets"][event_idx]
        for pmt in range(N_PMT):
            s = int(o_evt[pmt])
            e = int(o_evt[pmt + 1])
            t = np.asarray(t_evt[s:e], dtype=np.float32)
            a = np.asarray(a_evt[s:e], dtype=np.float32)
            n = min(len(t), max_points)
            lengths[pmt] = n
            if n > 0:
                t0, t1 = float(t[0]), float(t[-1])
                denom = max(t1 - t0, 1.0)
                wave[pmt, :n, 0] = (t[:n] - t0) / denom
                amax = max(float(np.max(np.abs(a[:n]))), 1.0)
                wave[pmt, :n, 1] = a[:n] / amax
        return wave, lengths

    z = data
    evt_off = int(z["event_offsets"][event_idx])
    for pmt in range(N_PMT):
        pmt_idx = evt_off + pmt
        s = int(z["pmt_offsets"][pmt_idx])
        e = int(z["pmt_offsets"][pmt_idx + 1])
        t = z["time_values"][s:e]
        a = z["adc_values"][s:e]
        n = min(len(t), max_points)
        lengths[pmt] = n
        if n > 0:
            t0, t1 = float(t[0]), float(t[-1])
            denom = max(t1 - t0, 1.0)
            wave[pmt, :n, 0] = (t[:n] - t0) / denom
            amax = max(float(np.max(np.abs(a[:n]))), 1.0)
            wave[pmt, :n, 1] = a[:n] / amax
    return wave, lengths


def load_wfsampling_file(
    entry: FileEntry,
    ws_root: str = WFSAMPLING_OUTPUT_ROOT,
):
    """Load one WFSampling file into memory. Returns (fmt, data) with fmt in {'npy','npz'}."""
    npy_path = os.path.join(ws_root, entry.cls_name, f"wfsampling_{entry.file_id}.npy")
    npz_path = os.path.join(ws_root, entry.cls_name, f"wfsampling_{entry.file_id}.npz")
    path = npy_path if os.path.isfile(npy_path) else npz_path
    if not os.path.isfile(path):
        raise FileNotFoundError(npy_path)
    if path.endswith(".npy"):
        return "npy", np.load(path, allow_pickle=True).item()
    return "npz", np.load(path, allow_pickle=False)


def load_wfsampling(
    entry: FileEntry,
    event_idx: int,
    max_points: int,
    ws_root: str = WFSAMPLING_OUTPUT_ROOT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return wave (N_PMT, max_points, 2) and lengths (N_PMT,). Channels: norm_time, norm_adc."""
    cached = load_wfsampling_file(entry, ws_root=ws_root)
    return wfsampling_event_from_file(cached, event_idx, max_points)


def estimate_waveform_params(
    entries,
    source: str,
    sample_events: int = 50,
    ws_root: str = WFSAMPLING_OUTPUT_ROOT,
) -> Dict:
    from pid_lib.data_io import load_aligned_file

    if source == "decon_waveform":
        max_len = 0
        seen = 0
        for entry in entries:
            paths = find_waveform_files(entry.root, entry.file_id)
            path = paths.get("decon_waveform")
            if not path:
                continue
            data = np.load(path, allow_pickle=False)
            n_ev = int(data["n_events"])
            for evt in range(min(n_ev, 3)):
                _, meta = reconstruct_event_from_compact_npz(path, evt)
                max_len = max(max_len, meta.get("n_samples", 0))
                seen += 1
                if seen >= sample_events:
                    break
            if seen >= sample_events:
                break
        return {"max_len": max(max_len, 200), "channels": 1}
    if source == "waveform":
        max_len = 0
        seen = 0
        for entry in entries:
            paths = find_waveform_files(entry.root, entry.file_id)
            path = paths.get("waveform")
            if not path:
                continue
            data = np.load(path, allow_pickle=False)
            n_ev = int(data["n_events"])
            for evt in range(min(n_ev, 3)):
                _, meta = reconstruct_event_from_compact_npz(path, evt)
                max_len = max(max_len, meta.get("n_samples", 0))
                seen += 1
                if seen >= sample_events:
                    break
            if seen >= sample_events:
                break
        return {"max_len": max(max_len, 200), "channels": 1}
    if source == "decon_npevst":
        max_pts = 0
        seen = 0
        for entry in entries:
            paths = find_waveform_files(entry.root, entry.file_id)
            path = paths.get("decon_npevst")
            if not path:
                continue
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception:
                continue
            n_ev = len(arr)
            for evt in range(min(n_ev, 2)):
                hits = arr[evt]
                if hits is None or len(hits) == 0:
                    seen += 1
                    continue
                pmt = hits[:, 2].astype(np.int64)
                pmt = pmt[(pmt >= 0) & (pmt < N_PMT)]
                if len(pmt):
                    cnt = np.bincount(pmt, minlength=N_PMT)
                    max_pts = max(max_pts, int(cnt.max()))
                seen += 1
                if seen >= sample_events:
                    break
            if seen >= sample_events:
                break
        return {"max_points": max(max_pts, 8), "channels": 2, "t_max": 1000.0}
    if source == "wfsampling":
        max_pts = 0
        seen = 0
        for entry in entries:
            _, _, meta = load_aligned_file(entry)
            n_ev = meta.get("n_kept", 1)
            npy_path = os.path.join(ws_root, entry.cls_name, f"wfsampling_{entry.file_id}.npy")
            npz_path = os.path.join(ws_root, entry.cls_name, f"wfsampling_{entry.file_id}.npz")
            path = npy_path if os.path.isfile(npy_path) else npz_path
            if not os.path.isfile(path):
                continue
            if path.endswith(".npy"):
                obj = np.load(path, allow_pickle=True).item()
                offsets = obj["offsets"]
                for i in range(min(n_ev, len(offsets), 2)):
                    off = offsets[i]
                    if isinstance(off, list):
                        lens = [off[j + 1] - off[j] for j in range(len(off) - 1)]
                    else:
                        lens = off[1:] - off[:-1]
                    if len(lens):
                        max_pts = max(max_pts, int(max(lens)))
                    seen += 1
                    if seen >= sample_events:
                        break
            else:
                z = np.load(path, allow_pickle=False)
                for i in range(min(n_ev, 2)):
                    evt_off = int(z["event_offsets"][i])
                    start = evt_off
                    end = start + N_PMT
                    po = z["pmt_offsets"][start:end + 1]
                    lens = po[1:] - po[:-1]
                    if len(lens):
                        max_pts = max(max_pts, int(np.max(lens)))
                    seen += 1
                    if seen >= sample_events:
                        break
            if seen >= sample_events:
                break
        return {"max_points": max(max_pts, 8), "channels": 2}
    raise ValueError(f"Unknown waveform source: {source}")
