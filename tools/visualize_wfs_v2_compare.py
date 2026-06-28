#!/usr/bin/env python3
"""Compare WFSampling v2 raw (WFS_wav_v2) vs decon (WFS_decon_wav_v2) on sample events.

Outputs under audit/viz/wfs_v2_compare/:
  - compare_{cls}_f{fid}_e{evt}_p{pmt}.png  : 2x3 panel (raw/decon waveform + WFS overlay)
  - density_{cls}_f{fid}_e{evt}.png         : per-PMT key-point count histogram
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_LABEL, CLASS_ROOTS, N_PMT
from tools.visualize_pid_data import reconstruct_decon_waveform, find_best_pmt, _load_npz
from WFSampling.show import load_raw_pmt, load_wfs_pmt, get_baseline_and_ac

WAV_V2 = "/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2"
DECON_V2 = "/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2"

DEFAULT_SAMPLES = [
    ("numu", 2000, 0),
    ("nue", 2574, 11),
    ("nc", 569, 3),
]


def load_decon_pmt(cls_name, file_id, event_idx, pmt_id):
    path = os.path.join(CLASS_ROOTS[cls_name], "waveform", f"decon_waveform_{file_id}.npz")
    if not os.path.isfile(path):
        return None, path
    dense = reconstruct_decon_waveform(path, event_idx)
    seg = dense[pmt_id]
    # decon stored as ADC in npz; WFS uses (ADC-1000)/100
    ac = (seg.astype(np.float32) - 1000.0) / 100.0
    return seg, path


def decon_ac_for_wfs(adc_segment):
    """Match WFSampling_v2 decon_mode AC: (adc-1000)/100, no clip to 0."""
    return (adc_segment.astype(np.float32) - 1000.0) / 100.0


def pmt_point_counts(wfs_root, cls_name, file_id, event_idx):
    from pid_lib.waveform_io import load_wfsampling_file
    from pid_lib.data_io import FileEntry

    entry = FileEntry(cls_name=cls_name, file_id=file_id, label=0, root=CLASS_ROOTS[cls_name])
    cached = load_wfsampling_file(entry, ws_root=wfs_root)
    _, data = cached
    off = data["offsets"][event_idx]
    if isinstance(off, list):
        return np.array([off[i + 1] - off[i] for i in range(len(off) - 1)], dtype=np.int32)
    return (off[1:] - off[:-1]).astype(np.int32)


def pick_pmt(cls_name, file_id, event_idx, pmt_hint=None):
    if pmt_hint is not None:
        return int(pmt_hint)
    wf_path = os.path.join(CLASS_ROOTS[cls_name], "waveform", f"decon_waveform_{file_id}.npz")
    if os.path.isfile(wf_path):
        dense = reconstruct_decon_waveform(wf_path, event_idx)
        return int(find_best_pmt(dense))
    counts = pmt_point_counts(WAV_V2, cls_name, file_id, event_idx)
    return int(np.argmax(counts))


def plot_density(cls_name, file_id, event_idx, output_dir):
    c_wav = pmt_point_counts(WAV_V2, cls_name, file_id, event_idx)
    c_dec = pmt_point_counts(DECON_V2, cls_name, file_id, event_idx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle(
        f"WFS v2 key-point counts per PMT | {cls_name} f{file_id} e{event_idx}",
        fontsize=11,
    )
    bins = np.arange(0, max(c_wav.max(), c_dec.max(), 10) + 2) - 0.5
    for ax, counts, title, color in [
        (axes[0], c_wav, f"raw WFS_wav_v2 (nz={(c_wav>0).mean()*100:.1f}%)", "steelblue"),
        (axes[1], c_dec, f"decon WFS_decon_wav_v2 (nz={(c_dec>0).mean()*100:.1f}%)", "darkorange"),
    ]:
        ax.hist(counts, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.median(counts), color="red", ls="--", lw=1, label=f"median={np.median(counts):.0f}")
        ax.axvline(np.percentile(counts, 95), color="green", ls=":", lw=1,
                   label=f"p95={np.percentile(counts,95):.0f}")
        ax.set_title(title)
        ax.set_xlabel("# key points / PMT")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("# PMTs")
    plt.tight_layout()
    out = os.path.join(output_dir, f"density_{cls_name}_f{file_id}_e{event_idx}.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved {out}")
    return {
        "wav_nz_frac": float((c_wav > 0).mean()),
        "decon_nz_frac": float((c_dec > 0).mean()),
        "wav_mean": float(c_wav.mean()),
        "decon_mean": float(c_dec.mean()),
        "wav_p95": float(np.percentile(c_wav, 95)),
        "decon_p95": float(np.percentile(c_dec, 95)),
    }


def plot_compare(cls_name, file_id, event_idx, pmt_id, output_dir):
    raw, raw_src = load_raw_pmt(cls_name, file_id, event_idx, pmt_id)
    decon_seg, decon_src = load_decon_pmt(cls_name, file_id, event_idx, pmt_id)
    t_w, a_w, pw = load_wfs_pmt(cls_name, file_id, event_idx, pmt_id, WAV_V2)
    t_d, a_d, pd = load_wfs_pmt(cls_name, file_id, event_idx, pmt_id, DECON_V2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"WFS v2 compare | {cls_name} (label={CLASS_LABEL[cls_name]})  "
        f"file={file_id} event={event_idx} PMT={pmt_id}",
        fontsize=12,
    )

    # Row 0: raw waveform branch
    ax = axes[0, 0]
    if raw is not None:
        ax.plot(np.arange(len(raw)), raw, lw=0.8, color="royalblue")
        ax.set_title(f"Raw waveform\n{os.path.basename(str(raw_src))}")
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("sample"); ax.set_ylabel("ADC (inv)"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ac_raw = None
    if raw is not None:
        ac_raw, bl = get_baseline_and_ac(raw, bl_len=100, thr_flat=15.0)
        ax.plot(np.arange(len(ac_raw)), ac_raw, color="lightgray", lw=0.8)
        ax.axhline(15.0, ls="--", color="blue", alpha=0.7, label="global_thr=15")
        ax.set_title(f"Raw AC (baseline={bl:.1f})")
        ax.legend(fontsize=7)
    ax.set_xlabel("sample"); ax.set_ylabel("AC"); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if ac_raw is not None:
        ax.plot(np.arange(len(ac_raw)), ac_raw, color="lightgray", lw=0.8, label="AC")
    if t_w is not None and len(t_w):
        ax.plot(t_w, a_w, "ro-", ms=4, lw=1, label=f"WFS_wav_v2 N={len(t_w)}")
    ax.set_title(f"raw WFS key points\nglobal_thr={pw.get('global_thr', 15)}")
    ax.legend(fontsize=7)
    ax.set_xlabel("time (ns)"); ax.set_ylabel("amp"); ax.grid(True, alpha=0.3)

    # Row 1: decon waveform branch
    ax = axes[1, 0]
    if decon_seg is not None:
        ax.plot(np.arange(len(decon_seg)), decon_seg, lw=0.8, color="teal")
        ax.set_title(f"Decon waveform (ADC)\n{os.path.basename(decon_src)}")
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("sample"); ax.set_ylabel("ADC"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ac_dec = None
    if decon_seg is not None:
        ac_dec = decon_ac_for_wfs(decon_seg)
        ax.plot(np.arange(len(ac_dec)), ac_dec, color="lightgray", lw=0.8)
        ax.axhline(1.0, ls="--", color="blue", alpha=0.7, label="global_thr=1")
        ax.set_title("Decon AC (WFS input)")
        ax.legend(fontsize=7)
    ax.set_xlabel("sample"); ax.set_ylabel("AC"); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if ac_dec is not None:
        ax.plot(np.arange(len(ac_dec)), ac_dec, color="lightgray", lw=0.8, label="AC")
    if t_d is not None and len(t_d):
        ax.plot(t_d, a_d, "o-", color="darkorange", ms=4, lw=1,
                label=f"WFS_decon_v2 N={len(t_d)}")
    elif t_d is not None:
        ax.text(0.5, 0.6, "no key points", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title(f"decon WFS key points\nglobal_thr={pd.get('global_thr', 1)}")
    ax.legend(fontsize=7)
    ax.set_xlabel("time (ns)"); ax.set_ylabel("amp"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, f"compare_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved {out}")
    return {
        "pmt": pmt_id,
        "wav_points": int(len(t_w)) if t_w is not None else 0,
        "decon_points": int(len(t_d)) if t_d is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "audit", "viz", "wfs_v2_compare"))
    parser.add_argument("--pmt-hints", nargs="*", type=int, default=None,
                        help="Optional PMT ids for numu nue nc in order")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hints = args.pmt_hints or [None, 14796, 14436]
    summary = []
    for (cls_name, fid, evt), pmt_hint in zip(DEFAULT_SAMPLES, hints):
        print(f"\n[{cls_name}] file={fid} event={evt}")
        pmt = pick_pmt(cls_name, fid, evt, pmt_hint=pmt_hint)
        dens = plot_density(cls_name, fid, evt, args.output_dir)
        cmp_meta = plot_compare(cls_name, fid, evt, pmt, args.output_dir)
        row = {"class": cls_name, "file_id": fid, "event": evt, **dens, **cmp_meta}
        summary.append(row)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary.json ({len(summary)} events)")


if __name__ == "__main__":
    main()
