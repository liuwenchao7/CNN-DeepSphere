#!/usr/bin/env python3
"""Cross-modal visualization for one PID event (det_fea, waveform, FHT, WFS).

Example:
  python3 tools/visualize_pid_event_suite.py --seed 42
  python3 tools/visualize_pid_event_suite.py --class numu --file-id 2000 --event 0
"""

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_LABEL, CLASS_ROOTS, N_PMT
from pid_lib.data_io import discover_valid_entries

# Reuse plot helpers from visualize_pid_data
from tools.visualize_pid_data import (
    plot_det_fea,
    plot_waveform_compare,
    reconstruct_decon_waveform,
    find_best_pmt,
    _load_npz,
)

WFS_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v1"
DECON_FHT_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav"
FHT_START = -20.0
FHT_WIDTH = 300


def load_fht_pmt(cls_name, file_id, event_idx, pmt_id):
    path = os.path.join(DECON_FHT_ROOT, cls_name, f"decon_waveform_{file_id}.npz")
    if not os.path.isfile(path):
        return None
    z = _load_npz(path)
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    copy = z["copyNo"][evt_off:evt_end].astype(np.int64)
    wf = z["waveform"][evt_off:evt_end].astype(np.float32)
    if pmt_id in copy:
        idx = int(np.where(copy == pmt_id)[0][0])
        return wf[idx]
    return np.zeros((FHT_WIDTH,), dtype=np.float32)


def load_full_decon_segment(cls_name, file_id, event_idx, pmt_id):
    wf_path = os.path.join(CLASS_ROOTS[cls_name], "waveform", f"decon_waveform_{file_id}.npz")
    if not os.path.isfile(wf_path):
        return None, "missing decon_waveform"
    dense = reconstruct_decon_waveform(wf_path, event_idx)
    return dense[pmt_id], wf_path


def load_fht_time_ns(cls_name, file_id, event_idx, pmt_id):
    fht_path = os.path.join(CLASS_ROOTS[cls_name], "elec_fea", f"x_fht_pmt_{file_id}.npy")
    if not os.path.isfile(fht_path):
        return None
    fht = np.load(fht_path, allow_pickle=True).astype(np.float32)
    if event_idx >= fht.shape[0]:
        return None
    return float(fht[event_idx, pmt_id])


def plot_fht_with_full_decon(cls_name, file_id, event_idx, pmt_id, output_dir):
    fht_seg = load_fht_pmt(cls_name, file_id, event_idx, pmt_id)
    full_seg, src = load_full_decon_segment(cls_name, file_id, event_idx, pmt_id)
    fht_ns = load_fht_time_ns(cls_name, file_id, event_idx, pmt_id)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"decon FHT window vs full decon  {cls_name} f{file_id} e{event_idx} PMT={pmt_id}",
        fontsize=11,
    )

    ax = axes[0]
    if fht_seg is not None:
        ax.plot(np.arange(len(fht_seg)), fht_seg, lw=0.9, color="darkorange")
        ax.set_title(f"decon_wav_fht [{FHT_START}, {FHT_START + FHT_WIDTH})")
    else:
        ax.text(0.5, 0.5, "no fht window", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Sample in window")
    ax.set_ylabel("ADC")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if full_seg is not None:
        ax.plot(np.arange(len(full_seg)), full_seg, lw=0.8, color="darkorange")
        if fht_ns is not None and fht_ns > 0:
            fht_idx = int(round(fht_ns))
            if 0 <= fht_idx < len(full_seg):
                ax.axvline(fht_idx, color="red", ls="--", lw=1.2, label=f"FHT={fht_ns:.1f}")
                ax.legend(fontsize=8)
        ax.set_title(f"full decon_waveform ({os.path.basename(src)})")
    else:
        ax.text(0.5, 0.5, src, ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("ADC")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, f"fht_full_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"  Saved {out}")
    return out


def plot_wfs_two_panel(cls_name, file_id, event_idx, pmt_id, output_dir, wfs_root=WFS_ROOT):
    from WFSampling.show import load_raw_pmt, load_wfs_pmt, get_baseline_and_ac

    raw, raw_src = load_raw_pmt(cls_name, file_id, event_idx, pmt_id)
    t_pts, a_pts, params = load_wfs_pmt(cls_name, file_id, event_idx, pmt_id, wfs_root)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"WFSampling | {cls_name} f{file_id} e{event_idx} PMT={pmt_id}",
        fontsize=11,
    )

    ax = axes[0]
    if raw is not None:
        ax.plot(np.arange(len(raw)), raw, lw=0.8, color="royalblue")
        ax.set_title(f"Raw waveform (inverted)\n{raw_src}")
    else:
        ax.text(0.5, 0.5, raw_src, ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Raw waveform")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("ADC")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if raw is not None:
        ac, _ = get_baseline_and_ac(
            raw,
            bl_len=int(params.get("bl_len", 100)),
            thr_flat=float(params.get("thr_flat", 15)),
        )
        ax.plot(np.arange(len(ac)), ac, color="lightgray", lw=0.8, label="AC")
    if t_pts is not None and len(t_pts) > 0:
        ax.plot(t_pts, a_pts, "ro-", ms=4, lw=1, label=f"key points N={len(t_pts)}")
    ax.set_title("Key points on AC")
    ax.legend(fontsize=8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, f"wfs_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"  Saved {out}")
    return out


def pick_random_events(seed=42):
    rng = random.Random(seed)
    picks = {}
    for cls_name in ("numu", "nue", "nc"):
        entries = [e for e in discover_valid_entries() if e.cls_name == cls_name]
        if not entries:
            continue
        entry = rng.choice(entries)
        y = np.load(entry.y_path, allow_pickle=True)
        n_evt = max(1, int(y.shape[0]))
        evt = rng.randint(0, n_evt - 1)
        picks[cls_name] = (entry.file_id, evt)
    return picks


def render_event_suite(cls_name, file_id, event_idx, output_dir, wfs_root=WFS_ROOT):
    root = CLASS_ROOTS[cls_name]
    os.makedirs(output_dir, exist_ok=True)

    # brightest PMT from decon for consistent pmt across panels
    wf_path = os.path.join(root, "waveform", f"decon_waveform_{file_id}.npz")
    pmt_id = 0
    if os.path.isfile(wf_path):
        dense = reconstruct_decon_waveform(wf_path, event_idx)
        pmt_id = find_best_pmt(dense)

    print(f"\n[{cls_name}] file={file_id} event={event_idx} pmt={pmt_id}")
    plot_det_fea(cls_name, root, file_id, event_idx, output_dir, pmt_hint=pmt_id)
    plot_waveform_compare(cls_name, root, file_id, event_idx, pmt_id, output_dir)
    plot_fht_with_full_decon(cls_name, file_id, event_idx, pmt_id, output_dir)
    plot_wfs_two_panel(cls_name, file_id, event_idx, pmt_id, output_dir, wfs_root=wfs_root)
    meta = {
        "class": cls_name,
        "label": CLASS_LABEL[cls_name],
        "file_id": file_id,
        "event": event_idx,
        "pmt": pmt_id,
    }
    with open(os.path.join(output_dir, f"suite_{cls_name}_f{file_id}_e{event_idx}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls_name", choices=list(CLASS_ROOTS.keys()))
    parser.add_argument("--file-id", type=int, default=None)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random pick one event per class when --class not set")
    parser.add_argument("--wfs-root", default=WFS_ROOT)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "audit", "viz", "event_suite"))
    args = parser.parse_args()

    if args.cls_name and args.file_id is not None:
        render_event_suite(args.cls_name, args.file_id, args.event, args.output_dir, args.wfs_root)
        return

    picks = pick_random_events(seed=args.seed)
    print(f"Random picks (seed={args.seed}): {picks}")
    for cls_name, (fid, evt) in picks.items():
        render_event_suite(cls_name, fid, evt, args.output_dir, wfs_root=args.wfs_root)


if __name__ == "__main__":
    main()
