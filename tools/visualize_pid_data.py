#!/usr/bin/env python3
"""EXECUTION_PLAN Task 3 – visualize det_fea and waveform data.

Usage examples:
  python3 tools/visualize_pid_data.py --class numu --file-id 2000 --event 0 --pmt 500
  python3 tools/visualize_pid_data.py --class nue  --file-id 2500 --event 0
  python3 tools/visualize_pid_data.py --class nc   --file-id 0    --event 0

Outputs (saved to --output-dir):
  1. det_fea_{cls}_f{id}_e{evt}.png   – time-nPE scatter for one event
  2. waveform_{cls}_f{id}_e{evt}_p{pmt}.png – side-by-side comparison of
       raw waveform / decon_waveform / decon_npevst for one PMT
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_LABEL, CLASS_ROOTS, N_PMT


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_npz(path):
    return np.load(path, allow_pickle=False)


def reconstruct_waveform_fixed(npz_path, event_idx, n_pmt=N_PMT):
    """Reconstruct dense (n_pmt, n_samples) from compact fixed-length waveform.npz."""
    z = _load_npz(npz_path)
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    copy = z["copyNo"][evt_off:evt_end].astype(np.int64)
    wf   = z["waveform"][evt_off:evt_end].astype(np.float32)   # (n_hit, 1008)
    n_samples = wf.shape[1]
    dense = np.zeros((n_pmt, n_samples), dtype=np.float32)
    dense[copy] = wf
    return dense


def reconstruct_decon_waveform(npz_path, event_idx, n_pmt=N_PMT):
    """Reconstruct dense (n_pmt, max_len) from ragged decon_waveform.npz."""
    z = _load_npz(npz_path)
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    copy    = z["copyNo"][evt_off:evt_end].astype(np.int64)
    wf_off  = z["waveform_offsets"][evt_off : evt_end + 1]
    wf_flat = z["waveform"]
    # Each PMT has variable-length waveform stored as contiguous uint16 values
    segments = []
    for i in range(len(copy)):
        s, e = int(wf_off[i]), int(wf_off[i + 1])
        seg = wf_flat[s:e].astype(np.float32)
        segments.append((int(copy[i]), seg))
    if not segments:
        return np.zeros((n_pmt, 1), dtype=np.float32)
    max_len = max(len(seg) for _, seg in segments)
    dense = np.zeros((n_pmt, max_len), dtype=np.float32)
    for pmt_id, seg in segments:
        dense[pmt_id, : len(seg)] = seg
    return dense


def load_decon_npevst_event(npy_path, event_idx):
    """Return (n_hits, 3) array: [npe, time, pmt_id] for one event."""
    try:
        if npy_path.endswith(".npz"):
            z = np.load(npy_path, allow_pickle=False)
            # Fallback layout support if future files switch to compact npz.
            if "events" in z:
                return z["events"][event_idx]
            raise ValueError("unsupported npz layout for decon_npevst")
        a = np.load(npy_path, allow_pickle=True)
        return a[event_idx]
    except Exception as exc:
        raise RuntimeError(f"failed to load decon_npevst ({exc})")


def find_best_pmt(dense, pmt_hint=None):
    """Return PMT index with largest total signal, or pmt_hint if given."""
    if pmt_hint is not None:
        return int(pmt_hint)
    total = np.sum(np.abs(dense), axis=1)
    return int(np.argmax(total))


# ── plot det_fea ─────────────────────────────────────────────────────────────

def plot_det_fea(cls_name, root, file_id, event_idx, output_dir, pmt_hint=None):
    import glob
    det_dir = os.path.join(root, "det_fea")
    candidates = glob.glob(os.path.join(det_dir, f"time_npe_{file_id}.npz"))
    if not candidates:
        print(f"[skip] det_fea not found: {det_dir}/time_npe_{file_id}.npz")
        return None

    z = _load_npz(candidates[0])
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    pmt_ids = z["pmtID"][evt_off:evt_end]

    # Collect (time, npe) per PMT for this event
    all_times, all_npes = [], []
    pmt_data = {}
    for i, pmt in enumerate(pmt_ids):
        s = int(z["pmt_offsets"][evt_off + i])
        e = int(z["pmt_offsets"][evt_off + i + 1])
        if e <= s:
            continue
        t_arr = z["time"][s:e]
        n_arr = z["npe"][s:e]
        all_times.append(t_arr)
        all_npes.append(n_arr)
        pmt_data[int(pmt)] = (t_arr, n_arr)

    # Choose a PMT for the single-PMT subplot (brightest if not given)
    if pmt_hint is not None and pmt_hint in pmt_data:
        focus_pmt = pmt_hint
    else:
        focus_pmt = max(pmt_data.keys(),
                        key=lambda p: float(pmt_data[p][1].sum())) if pmt_data else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"det_fea  {cls_name}  file={file_id}  event={event_idx}  "
                 f"n_hit_pmt={len(pmt_ids)}", fontsize=11)

    # ── Left: all PMTs scatter ─────────────────────────────────────────────
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(pmt_ids), 1)))
    for i, (ta, na) in enumerate(zip(all_times, all_npes)):
        ax.scatter(ta, na, s=5, alpha=0.4, color=colors[i % len(colors)])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("nPE")
    ax.set_title("All hit PMTs (each color = one PMT)")
    ax.grid(True, alpha=0.3)

    # ── Right: single PMT detail ───────────────────────────────────────────
    ax = axes[1]
    if focus_pmt is not None and focus_pmt in pmt_data:
        t_f, n_f = pmt_data[focus_pmt]
        ax.stem(t_f, n_f, linefmt="C0-", markerfmt="C0o", basefmt=" ")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("nPE")
        ax.set_title(f"Single PMT #{focus_pmt}  (total nPE={n_f.sum():.1f})")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Single PMT detail")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"det_fea_{cls_name}_f{file_id}_e{event_idx}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"  Saved {out}")
    return out


# ── plot waveform comparison ──────────────────────────────────────────────────

def plot_waveform_compare(cls_name, root, file_id, event_idx, pmt_hint,
                          output_dir):
    wf_dir = os.path.join(root, "waveform")

    raw_path    = os.path.join(wf_dir, f"waveform_{file_id}.npz")
    decon_path  = os.path.join(wf_dir, f"decon_waveform_{file_id}.npz")
    npevst_path = os.path.join(wf_dir, f"decon_npevst_{file_id}.npy")

    raw_dense = decon_dense = npevst_data = None
    notes = {}
    npevst_fallback = None

    if os.path.isfile(raw_path):
        try:
            raw_dense = reconstruct_waveform_fixed(raw_path, event_idx)
            # Invert 14-bit FADC: physical signal = 16384 - ADC
            raw_dense = 16384.0 - raw_dense
            notes["raw"] = f"inverted 14-bit ADC, shape {raw_dense.shape}"
        except Exception as exc:
            notes["raw"] = f"ERROR: {exc}"
    else:
        notes["raw"] = "file missing"

    if os.path.isfile(decon_path):
        try:
            decon_dense = reconstruct_decon_waveform(decon_path, event_idx)
            notes["decon"] = f"shape {decon_dense.shape}"
        except Exception as exc:
            notes["decon"] = f"ERROR: {exc}"
    else:
        notes["decon"] = "file missing"

    if os.path.isfile(npevst_path):
        try:
            npevst_data = load_decon_npevst_event(npevst_path, event_idx)
            notes["npevst"] = f"hits={npevst_data.shape[0]} cols=[npe,time,pmt]"
        except Exception as exc:
            notes["npevst"] = f"ERROR: {exc}"
            if decon_dense is not None:
                # File is present but unreadable (e.g. truncated pickle); for
                # visualization only, synthesize sparse hits from decon waveform.
                npevst_fallback = "from_decon_waveform"
    else:
        notes["npevst"] = "file missing"

    # Choose PMT to display
    pmt_id = find_best_pmt(raw_dense if raw_dense is not None else decon_dense,
                           pmt_hint)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{cls_name}  file={file_id}  event={event_idx}  "
                 f"PMT={pmt_id}  (label={CLASS_LABEL[cls_name]})", fontsize=11)

    # --- subplot 1: raw waveform ---
    ax = axes[0]
    if raw_dense is not None:
        wave = raw_dense[pmt_id]
        ax.plot(np.arange(len(wave)), wave, lw=0.8, color="royalblue")
        ax.set_title(f"waveform.npz\n{notes['raw'][:60]}")
    else:
        ax.text(0.5, 0.5, notes["raw"], ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("waveform.npz")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (ADC)")
    ax.grid(True, alpha=0.3)

    # --- subplot 2: decon waveform ---
    ax = axes[1]
    if decon_dense is not None:
        wave = decon_dense[pmt_id]
        ax.plot(np.arange(len(wave)), wave, lw=0.8, color="darkorange")
        ax.set_title(f"decon_waveform.npz\n{notes['decon'][:60]}")
    else:
        ax.text(0.5, 0.5, notes["decon"], ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("decon_waveform.npz")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (ADC)")
    ax.grid(True, alpha=0.3)

    # --- subplot 3: decon_npevst ---
    ax = axes[2]
    if npevst_data is not None and npevst_data.ndim == 2 and npevst_data.shape[1] >= 3:
        # columns: npe, time, pmt_id
        mask = (npevst_data[:, 2].astype(int) == pmt_id)
        hits = npevst_data[mask]
        if len(hits):
            ax.stem(hits[:, 1], hits[:, 0],
                    linefmt="g-", markerfmt="go", basefmt=" ")
            ax.set_title(f"decon_npevst.npy\n"
                         f"hits for PMT={pmt_id}: {len(hits)}")
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id}: no hits",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"decon_npevst.npy ({notes['npevst'][:40]})")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("nPE")
    elif npevst_fallback == "from_decon_waveform" and decon_dense is not None:
        wf = decon_dense[pmt_id]
        idx = np.where(np.abs(wf) > 0)[0]
        if len(idx):
            ax.stem(idx.astype(np.float32), wf[idx],
                    linefmt="g-", markerfmt="go", basefmt=" ")
            ax.set_title("decon_npevst.npy unreadable\nfallback: from decon_waveform")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("ADC")
        else:
            ax.text(0.5, 0.5, "fallback: no non-zero bins",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("decon_npevst.npy unreadable")
    else:
        ax.text(0.5, 0.5, notes.get("npevst", "N/A"),
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("decon_npevst.npy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"waveform_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"  Saved {out}")
    return out


# ── plot FHT window waveforms ───────────────────────────────────────────────

def reconstruct_fht_event(npz_path, event_idx, n_pmt=N_PMT):
    z = _load_npz(npz_path)
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    copy = z["copyNo"][evt_off:evt_end].astype(np.int64)
    wf = z["waveform"][evt_off:evt_end].astype(np.float32)
    width = wf.shape[1]
    dense = np.zeros((n_pmt, width), dtype=np.float32)
    if len(copy):
        dense[copy] = wf
    return dense


def plot_fht_compare(cls_name, file_id, event_idx, pmt_hint, output_dir,
                     decon_root="/disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav",
                     wav_root="/disk_pool1/liuwc/data/cnn+ds/pid/wav_fht+-/wav",
                     window_start=-20.0, window_width=300):
    decon_path = os.path.join(decon_root, cls_name, f"decon_waveform_{file_id}.npz")
    wav_path = os.path.join(wav_root, cls_name, f"waveform_{file_id}.npz")

    decon_dense = wav_dense = None
    notes = {}
    for kind, path in [("decon", decon_path), ("wav", wav_path)]:
        if not os.path.isfile(path):
            notes[kind] = "file missing"
            continue
        try:
            dense = reconstruct_fht_event(path, event_idx)
            if kind == "decon":
                decon_dense = dense
            else:
                wav_dense = dense
            notes[kind] = f"shape {dense.shape}"
        except Exception as exc:
            notes[kind] = f"ERROR: {exc}"

    ref = wav_dense if wav_dense is not None else decon_dense
    if ref is None:
        print(f"[skip] no FHT files for {cls_name} file={file_id}")
        return None
    pmt_id = find_best_pmt(ref, pmt_hint)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"FHT window [{window_start}, {window_start + window_width}) ns  "
        f"{cls_name} file={file_id} event={event_idx} PMT={pmt_id}",
        fontsize=11,
    )

    ax = axes[0]
    if decon_dense is not None:
        wave = decon_dense[pmt_id]
        ax.plot(np.arange(len(wave)), wave, lw=0.9, color="darkorange")
        ax.set_title(f"decon_wav_fht\n{notes.get('decon', '')[:50]}")
    else:
        ax.text(0.5, 0.5, notes.get("decon", "missing"), ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("decon_wav_fht")
    ax.set_xlabel(f"Sample (window {window_width})")
    ax.set_ylabel("ADC")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if wav_dense is not None:
        wave = wav_dense[pmt_id]
        ax.plot(np.arange(len(wave)), wave, lw=0.9, color="royalblue")
        ax.set_title(f"wav_fht (16384-ADC inverted)\n{notes.get('wav', '')[:50]}")
    else:
        ax.text(0.5, 0.5, notes.get("wav", "missing"), ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("wav_fht")
    ax.set_xlabel(f"Sample (window {window_width})")
    ax.set_ylabel("ADC")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"fht_{cls_name}_f{file_id}_e{event_idx}_p{pmt_id}.png")
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"  Saved {out}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls_name",
                        choices=list(CLASS_ROOTS.keys()), required=True)
    parser.add_argument("--file-id", type=int, required=True)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--pmt", type=int, default=None,
                        help="PMT index to display; auto-selects brightest if omitted")
    parser.add_argument("--output-dir",
                        default=os.path.join(ROOT, "audit", "viz"))
    parser.add_argument("--fht-only", action="store_true",
                        help="Only plot FHT-window waveforms (decon_wav_fht vs wav_fht)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    root = CLASS_ROOTS[args.cls_name]

    print(f"\n[{args.cls_name}] file={args.file_id} event={args.event}")
    if args.fht_only:
        plot_fht_compare(args.cls_name, args.file_id, args.event, args.pmt, args.output_dir)
        return
    plot_det_fea(args.cls_name, root, args.file_id,
                 args.event, args.output_dir, pmt_hint=args.pmt)
    plot_waveform_compare(args.cls_name, root, args.file_id,
                          args.event, args.pmt, args.output_dir)
    plot_fht_compare(args.cls_name, args.file_id, args.event, args.pmt, args.output_dir)


if __name__ == "__main__":
    main()
