#!/usr/bin/env python3
"""EXECUTION_PLAN Task 4 – visualize WFSampling results.

For each of numu/nue/nc: shows raw waveform, AC waveform, and extracted key
points side-by-side for a chosen PMT.

Usage:
  python3 WFSampling/show.py --class nc   --file-id 0   --event 0 --pmt 0
  python3 WFSampling/show.py --class numu --file-id 2000 --event 0
  python3 WFSampling/show.py --class nue  --file-id 2500 --event 0
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

WFS_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling"


# ── signal processing (same parameters as WFSampling.py) ─────────────────────

def get_baseline_and_ac(adc, bl_len=100, thr_flat=15.0):
    adc = adc.astype(np.float32)
    piece = 8; flag_good = False; start = 0
    for start in range(0, bl_len - 3 * piece, piece):
        seg = adc[start:start + piece]; bl = np.mean(seg)
        if np.all(np.abs(seg - bl) <= thr_flat):
            flag_good = True; break
    if not flag_good:
        baseline_val = np.mean(adc[:bl_len])
    else:
        bl = np.mean(adc[start:start + piece]); end = bl_len
        for i in range(start, bl_len):
            if abs(adc[i] - bl) > thr_flat: end = i; break
        baseline_val = np.mean(adc[start:end])
    return np.clip(adc - baseline_val, 0, None), float(baseline_val)


# ── load raw waveform for one PMT ─────────────────────────────────────────────

def load_raw_pmt(cls_name, file_id, event_idx, pmt_id):
    wf_dir = os.path.join(CLASS_ROOTS[cls_name], "waveform")
    for name in (f"waveform_{file_id}.npz", f"waveform_{file_id}_compact.npz"):
        p = os.path.join(wf_dir, name)
        if not os.path.isfile(p):
            continue
        z = np.load(p, allow_pickle=False)
        s = int(z["event_offsets"][event_idx])
        e = int(z["event_offsets"][event_idx + 1])
        copy = z["copyNo"][s:e].astype(np.int64)
        wf   = z["waveform"][s:e].astype(np.float32)
        if pmt_id in copy:
            idx = int(np.where(copy == pmt_id)[0][0])
            return 16384.0 - wf[idx], name
        # PMT not among fired ones → return zeros
        n_samples = int(z.get("n_samples", wf.shape[1]))
        return np.zeros(n_samples, np.float32), name + " (PMT not fired)"
    return None, "waveform file not found"


# ── load WFSampling key points for one PMT ────────────────────────────────────

def load_wfs_pmt(cls_name, file_id, event_idx, pmt_id, wfs_root=WFS_ROOT):
    npy_path = os.path.join(wfs_root, cls_name, f"wfsampling_{file_id}.npy")
    npz_path = os.path.join(wfs_root, cls_name, f"wfsampling_{file_id}.npz")
    wfs_path = npy_path if os.path.isfile(npy_path) else npz_path
    if not os.path.isfile(wfs_path):
        return None, None, f"not found: {npy_path}"

    if wfs_path.endswith(".npy"):
        obj = np.load(wfs_path, allow_pickle=True).item()
        t_evt = obj["time"][event_idx]
        a_evt = obj["adc"][event_idx]
        o_evt = obj["offsets"][event_idx]
        s = int(o_evt[pmt_id]); e = int(o_evt[pmt_id + 1])
        t = t_evt[s:e]
        a = a_evt[s:e]
    else:
        z = np.load(wfs_path, allow_pickle=False)
        n_pmt = int(z["n_pmt"]) if "n_pmt" in z else N_PMT
        eo    = z["event_offsets"]
        po    = z["pmt_offsets"]
        base  = int(eo[event_idx]) * n_pmt + pmt_id
        s = int(po[base]);  e = int(po[base + 1])
        t = z["time_values"][s:e]
        a = z["adc_values"][s:e]
    # load params for title
    params_path = os.path.join(wfs_root, cls_name, "sampling_params.json")
    params = {}
    if os.path.isfile(params_path):
        with open(params_path) as f:
            params = json.load(f)
    return t, a, params


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_wfs(cls_name, file_id, event_idx, pmt_id, output_dir, wfs_root=WFS_ROOT):
    raw, raw_src = load_raw_pmt(cls_name, file_id, event_idx, pmt_id)
    t_pts, a_pts, params = load_wfs_pmt(cls_name, file_id, event_idx,
                                        pmt_id, wfs_root)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"WFSampling | {cls_name} (label={CLASS_LABEL[cls_name]})  "
        f"file={file_id}  event={event_idx}  PMT={pmt_id}\n"
        f"params: bl_len={params.get('bl_len',100)}  "
        f"global_thr={params.get('global_thr',15)}  "
        f"h_thr={params.get('h_thr',30)}  w_thr={params.get('w_thr',10)}",
        fontsize=10,
    )

    # ── (1) Raw waveform ──────────────────────────────────────────────────────
    ax = axes[0]
    if raw is not None:
        ax.plot(np.arange(len(raw)), raw, lw=0.8, color="royalblue")
        ax.set_title(f"Raw waveform\n({raw_src})")
    else:
        ax.text(0.5, 0.5, raw_src, ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Raw waveform (MISSING)")
    ax.set_xlabel("Sample index"); ax.set_ylabel("ADC (inverted)")
    ax.grid(True, alpha=0.3)

    # ── (2) AC waveform (baseline subtracted, clipped to ≥0) ─────────────────
    ax = axes[1]
    if raw is not None:
        ac, bl = get_baseline_and_ac(raw,
                                     bl_len=int(params.get("bl_len", 100)),
                                     thr_flat=float(params.get("thr_flat", 15)))
        ax.plot(np.arange(len(ac)), ac, color="lightgray", lw=0.8,
                label="AC waveform")
        ax.axhline(float(params.get("global_thr", 15)), ls="--",
                   color="blue", alpha=0.7, label=f"thr={params.get('global_thr',15)}")
        ax.set_title(f"AC waveform (baseline={bl:.1f})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no raw data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # ── (3) Key points overlay ────────────────────────────────────────────────
    ax = axes[2]
    if raw is not None:
        ax.plot(np.arange(len(ac)), ac, color="lightgray", lw=0.8,
                label="AC waveform")
    if t_pts is not None and len(t_pts) > 0:
        ax.plot(t_pts, a_pts, "ro-", ms=4, lw=1,
                label=f"Key points (N={len(t_pts)})")
        ax.set_title(f"Extracted key points (N={len(t_pts)})")
    elif t_pts is not None:
        ax.set_title("No key points extracted")
    else:
        ax.set_title(f"WFSampling not available\n{params}")
    if t_pts is not None and len(t_pts) == 0 and raw is not None:
        ax.text(0.5, 0.7, "below threshold", ha="center", va="center",
                transform=ax.transAxes, color="red")
    ax.legend(fontsize=8)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir,
                       f"show_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls_name",
                        choices=list(CLASS_ROOTS.keys()), required=True)
    parser.add_argument("--file-id", type=int, required=True)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--pmt", type=int, default=0)
    parser.add_argument("--wfs-root", default=WFS_ROOT)
    parser.add_argument("--output-dir",
                        default=os.path.join(ROOT, "audit", "viz"))
    args = parser.parse_args()
    plot_wfs(args.cls_name, args.file_id, args.event, args.pmt,
             args.output_dir, args.wfs_root)


if __name__ == "__main__":
    main()
