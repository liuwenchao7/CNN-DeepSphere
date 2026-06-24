#!/usr/bin/env python3
"""Cut waveform windows around FHT and save compact PID waveform files.

Output layout:
  <out-root>/<cls_name>/<prefix>_<file_id>.npz

Saved NPZ keys:
  - waveform: (n_hits_total, window) float32
  - copyNo: (n_hits_total,) int32
  - event_offsets: (n_events+1,) int64
  - n_events: scalar int
  - n_samples: scalar int (=window)
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_ROOTS
from pid_lib.data_io import discover_valid_entries, find_waveform_files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["decon_waveform", "waveform"])
    p.add_argument("--classes", nargs="+", default=["numu", "nue", "nc"])
    p.add_argument("--start", type=float, default=-20.0)
    p.add_argument("--width", type=int, default=300)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--out-root",
        default="/disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav",
        help="base output dir (class subdirs will be created)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _one_file(task):
    cls_name, file_id, kind, start, width, out_root, overwrite = task
    root = CLASS_ROOTS[cls_name]
    out_dir = os.path.join(out_root, cls_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{kind}_{file_id}.npz")
    if os.path.isfile(out_path) and not overwrite:
        return cls_name, file_id, "skip", 0

    fht_path = os.path.join(root, "elec_fea", f"x_fht_pmt_{file_id}.npy")
    if not os.path.isfile(fht_path):
        return cls_name, file_id, "no_fht", 0
    fht = np.load(fht_path, allow_pickle=True).astype(np.float32)

    src = find_waveform_files(root, file_id).get(kind)
    if not src:
        return cls_name, file_id, "no_src", 0

    try:
        z = np.load(src, allow_pickle=False)
        n_events = int(z["n_events"])
    except Exception:
        return cls_name, file_id, "bad_src", 0

    n_events = min(n_events, fht.shape[0])
    all_copy = []
    all_wave = []
    event_offsets = [0]
    kept_events = 0

    has_ragged = "waveform_offsets" in z

    for evt in range(n_events):
        s_evt = int(z["event_offsets"][evt])
        e_evt = int(z["event_offsets"][evt + 1])
        if e_evt <= s_evt:
            event_offsets.append(event_offsets[-1])
            continue

        hit_pmts = z["copyNo"][s_evt:e_evt].astype(np.int32)
        if len(hit_pmts) == 0:
            event_offsets.append(event_offsets[-1])
            continue

        start_idx = np.round(fht[evt, hit_pmts] + start).astype(np.int64)
        slices = np.zeros((len(hit_pmts), width), dtype=np.float32)

        if has_ragged:
            wf_off = z["waveform_offsets"][s_evt : e_evt + 1]
            wf_flat = z["waveform"]
            for i in range(len(hit_pmts)):
                ws = int(wf_off[i])
                we = int(wf_off[i + 1])
                raw = wf_flat[ws:we].astype(np.float32)
                seg = raw[1:] if len(raw) > 1 else np.zeros((0,), dtype=np.float32)
                if kind == "waveform":
                    seg = 16384.0 - seg
                full_len = len(seg)
                if full_len == 0:
                    continue
                st = int(np.clip(start_idx[i], 0, max(0, full_len - width)))
                en = min(st + width, full_len)
                slices[i, : en - st] = seg[st:en]
        else:
            wave_evt = z["waveform"][s_evt:e_evt].astype(np.float32)
            if kind == "waveform":
                wave_evt = 16384.0 - wave_evt
            full_len = wave_evt.shape[1]
            st = np.clip(start_idx, 0, max(0, full_len - width)).astype(np.int64)
            cols = st[:, None] + np.arange(width, dtype=np.int64)[None, :]
            cols = np.clip(cols, 0, full_len - 1)
            slices[:, :] = wave_evt[np.arange(len(hit_pmts))[:, None], cols]

        all_copy.append(hit_pmts)
        all_wave.append(slices)
        event_offsets.append(event_offsets[-1] + len(hit_pmts))
        kept_events += 1

    if all_wave:
        copy_no = np.concatenate(all_copy).astype(np.int32)
        waveform = np.concatenate(all_wave).astype(np.float32)
    else:
        copy_no = np.zeros((0,), dtype=np.int32)
        waveform = np.zeros((0, width), dtype=np.float32)

    np.savez_compressed(
        out_path,
        waveform=waveform,
        copyNo=copy_no,
        event_offsets=np.asarray(event_offsets, dtype=np.int64),
        n_events=np.asarray(n_events, dtype=np.int64),
        n_samples=np.asarray(width, dtype=np.int64),
    )
    return cls_name, file_id, "ok", kept_events


def main():
    args = parse_args()
    entries = discover_valid_entries()
    wanted = {(e.cls_name, e.file_id) for e in entries if e.cls_name in set(args.classes)}

    tasks = [
        (cls_name, fid, args.kind, args.start, args.width, args.out_root, args.overwrite)
        for cls_name, fid in sorted(wanted)
    ]
    print(f"[preprocess] kind={args.kind} files={len(tasks)} out={args.out_root}")

    ok = 0
    skip = 0
    fail = 0
    done = 0
    with Pool(args.workers) as pool:
        for cls_name, fid, status, kept in pool.imap_unordered(_one_file, tasks, chunksize=2):
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[{status}] {cls_name} file={fid}")
            if done % 50 == 0:
                print(f"[progress] {done}/{len(tasks)} ok={ok} skip={skip} fail={fail}")
    print(f"[summary] ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
