#!/usr/bin/env python3
"""Export visE aligned to train_cnn_ds_pid test predictions (drops unloadable WFS events)."""

import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.data_io import load_y_arrays, align_event_indices
from pid_lib.splits import load_manifest
from pid_lib.waveform_io import load_wfsampling_file, wfsampling_event_from_file


def build_test_index(entries, seed=42):
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    index = []
    skipped = 0
    for entry in shuffled:
        try:
            y, y_pmt = load_y_arrays(entry)
            if y_pmt is not None and y_pmt.shape[0] != y.shape[0]:
                n_kept = len(align_event_indices(y, y_pmt))
            else:
                n_kept = int(y.shape[0])
        except Exception:
            skipped += 1
            continue
        for evt in range(n_kept):
            index.append((entry, evt))
    print(f"[index] events={len(index)} skipped_files={skipped}")
    return index


def export_vise_wfsampling(manifest_dir, output_path, seed=42, max_wf_pts=256,
                           wfs_root="/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling"):
    test_e = load_manifest(os.path.join(manifest_dir, "manifest_test.json"))
    test_index = build_test_index(test_e, seed=seed)

    wfs_cache = {}
    vise = []
    dropped = 0
    for entry, evt in test_index:
        key = (entry.cls_name, entry.file_id)
        if key not in wfs_cache:
            try:
                wfs_cache[key] = load_wfsampling_file(entry, ws_root=wfs_root)
            except Exception:
                wfs_cache[key] = None
        cached = wfs_cache[key]
        if cached is None:
            dropped += 1
            continue
        try:
            wfsampling_event_from_file(cached, evt, max_wf_pts)
        except Exception:
            dropped += 1
            continue
        y, y_pmt = load_y_arrays(entry)
        if y_pmt is not None and y_pmt.shape[0] != y.shape[0]:
            y = y[align_event_indices(y, y_pmt)]
        vise.append(float(y[evt, 5]))

    print(f"[export] kept={len(vise)} dropped_waveform={dropped} index={len(test_index)}")
    vise = np.array(vise, dtype=np.float32)
    np.save(output_path, vise)
    print(f"Saved -> {output_path}")
    return vise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", default=os.path.join(ROOT, "outputs", "fea6"))
    p.add_argument("--wfs-root", default="/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export_vise_wfsampling(args.manifest_dir, args.output, seed=args.seed, wfs_root=args.wfs_root)


if __name__ == "__main__":
    main()
