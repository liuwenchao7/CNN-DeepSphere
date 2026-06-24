#!/usr/bin/env python3
"""EXECUTION_PLAN Task 4 – WFSampling for all three PID classes.

Reads compact waveform_*.npz (format: event_offsets, copyNo, waveform),
extracts baseline-subtracted key points per PMT, saves one `.npy` per input
file (faster write than compressed npz for full production).

Usage:
  python3 WFSampling/WFSampling.py --class nc   [--file-ids 0 1 2] [--workers 4]
  python3 WFSampling/WFSampling.py --class numu --max-files 10 --workers 4
  python3 WFSampling/WFSampling.py --class nue  --workers 4
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_ROOTS, N_PMT


OUTPUT_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling"


# ── signal processing ─────────────────────────────────────────────────────────

def get_baseline_and_ac(adc, bl_len=100, thr_flat=15.0):
    adc = adc.astype(np.float32)
    flag_good = False
    piece = 8
    start = 0
    for start in range(0, bl_len - 3 * piece, piece):
        seg = adc[start:start + piece]
        bl = np.mean(seg)
        if np.all(np.abs(seg - bl) <= thr_flat):
            flag_good = True
            break
    if not flag_good:
        baseline_val = np.mean(adc[:bl_len])
    else:
        bl = np.mean(adc[start:start + piece])
        end = bl_len
        for i in range(start, bl_len):
            if abs(adc[i] - bl) > thr_flat:
                end = i
                break
        baseline_val = np.mean(adc[start:end])
    AC = np.clip(adc - baseline_val, 0, None)
    return AC, float(baseline_val)


def extract_key_points(AC, global_thr=15.0, h_thr=30.0, w_thr=10, rise_step=2):
    if np.max(AC) <= global_thr:
        return np.array([], np.float32), np.array([], np.float32)
    out_t, out_a = [], []
    n = len(AC)
    in_pulse = looking = False
    last_vt = last_va = 0.0
    lmax_a = -1e9; lmax_t = -1
    lmin_a = 1e9;  lmin_t = -1
    conf_pt = conf_pa = -1.0
    first_sampled = False

    def add(t, v):
        if not out_t or t > out_t[-1]:
            out_t.append(float(t)); out_a.append(float(v))

    for i in range(1, n - 1):
        c = AC[i]
        if c > global_thr:
            if not in_pulse:
                in_pulse = looking = True
                last_vt = i; last_va = c
                lmax_a = lmin_a = c; lmax_t = lmin_t = i
            if c > lmax_a: lmax_a = c; lmax_t = i
            if c < lmin_a: lmin_a = c; lmin_t = i
            if looking:
                if c < lmax_a - h_thr:
                    conf_pt = lmax_t; conf_pa = lmax_a
                    looking = False; lmin_a = c; lmin_t = i
            else:
                if c > lmin_a + h_thr:
                    w = lmin_t - last_vt
                    if w >= w_thr:
                        add(last_vt, last_va)
                        if not first_sampled:
                            for tt in range(int(last_vt) + rise_step,
                                           int(conf_pt), rise_step):
                                add(tt, AC[tt])
                            first_sampled = True
                        add(conf_pt, conf_pa); add(lmin_t, lmin_a)
                        last_vt = lmin_t; last_va = lmin_a
                        lmax_a = c; lmax_t = i
                    looking = True
        else:
            if in_pulse:
                in_pulse = False
                if not looking:
                    w = i - last_vt
                    if w >= w_thr:
                        add(last_vt, last_va)
                        if not first_sampled:
                            for tt in range(int(last_vt) + rise_step,
                                           int(conf_pt), rise_step):
                                add(tt, AC[tt])
                            first_sampled = True
                        add(conf_pt, conf_pa); add(i, c)
                else:
                    if lmax_a - last_va >= h_thr:
                        w = i - last_vt
                        if w >= w_thr:
                            add(last_vt, last_va)
                            if not first_sampled:
                                for tt in range(int(last_vt) + rise_step,
                                               int(lmax_t), rise_step):
                                    add(tt, AC[tt])
                                first_sampled = True
                            add(lmax_t, lmax_a); add(i, c)
    return np.array(out_t, np.float32), np.array(out_a, np.float32)


# ── file-level processing ─────────────────────────────────────────────────────

def _load_waveform_events(wf_path, waveform_kind="waveform"):
    """Return list of dense (N_PMT, T) float32 arrays, one per event."""
    z = np.load(wf_path, allow_pickle=False)
    n_events = int(z["n_events"])

    if "waveform_offsets" in z.files:
        events = []
        wf_flat = z["waveform"]
        for evt in range(n_events):
            evt_off = int(z["event_offsets"][evt])
            evt_end = int(z["event_offsets"][evt + 1])
            copy = z["copyNo"][evt_off:evt_end].astype(np.int64)
            wf_off = z["waveform_offsets"][evt_off : evt_end + 1]
            max_len = 1
            segments = []
            for i in range(len(copy)):
                s = int(wf_off[i])
                e = int(wf_off[i + 1])
                if e > s + 1:
                    seg = wf_flat[s + 1 : e].astype(np.float32)
                    segments.append((int(copy[i]), seg))
                    max_len = max(max_len, len(seg))
            dense = np.zeros((N_PMT, max_len), dtype=np.float32)
            for pmt, seg in segments:
                dense[pmt, :len(seg)] = seg
            events.append(dense)
        return events

    events = []
    for evt in range(n_events):
        s = int(z["event_offsets"][evt])
        e = int(z["event_offsets"][evt + 1])
        copy = z["copyNo"][s:e].astype(np.int64)
        wf = z["waveform"][s:e].astype(np.float32)
        dense = np.zeros((N_PMT, wf.shape[1]), dtype=np.float32)
        dense[copy] = wf
        events.append(dense)
    return events


def _file_id_from_path(path, waveform_kind):
    base = os.path.basename(path).split(".")[0]
    if waveform_kind == "decon_waveform":
        return int(base.rsplit("_", 1)[-1])
    return int(base.split("_")[1])


def _find_waveform_path(input_root, file_id, waveform_kind):
    wf_dir = os.path.join(input_root, "waveform")
    if waveform_kind == "decon_waveform":
        names = (f"decon_waveform_{file_id}.npz",)
    else:
        names = (f"waveform_{file_id}.npz", f"waveform_{file_id}_compact.npz")
    for name in names:
        p = os.path.join(wf_dir, name)
        if os.path.isfile(p):
            return p
    return None


def process_file(cls_name, file_id, params, output_root, input_root, waveform_kind="waveform"):
    out_dir = os.path.join(output_root, cls_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wfsampling_{file_id}.npy")

    # Skip already-done files
    if os.path.isfile(out_path):
        try:
            obj = np.load(out_path, allow_pickle=True).item()
            if int(obj.get("file_id", -1)) == file_id and int(obj.get("n_events", 0)) > 0:
                return "skipped", out_path
        except Exception:
            pass

    wf_path = _find_waveform_path(input_root, file_id, waveform_kind)
    if wf_path is None:
        return "missing", f"no {waveform_kind} for {cls_name}/{file_id}"

    try:
        events = _load_waveform_events(wf_path, waveform_kind)
        n_events = len(events)

        event_time = []
        event_adc = []
        event_offsets = []

        empty_count = 0
        for dense in events:
            all_t, all_a = [], []
            pmt_offsets = [0]  # per-event PMT offsets, length N_PMT+1
            invert = waveform_kind == "waveform"
            for pmt in range(N_PMT):
                raw = dense[pmt]
                signal = (16384.0 - raw) if invert else raw
                AC, _ = get_baseline_and_ac(
                    signal,
                    bl_len=params["bl_len"],
                    thr_flat=params["thr_flat"],
                )
                t_pts, a_pts = extract_key_points(
                    AC,
                    global_thr=params["global_thr"],
                    h_thr=params["h_thr"],
                    w_thr=params["w_thr"],
                    rise_step=params["rise_step"],
                )
                if len(t_pts) == 0:
                    empty_count += 1
                all_t.extend(t_pts.tolist())
                all_a.extend(a_pts.tolist())
                pmt_offsets.append(len(all_t))
            event_time.append(np.array(all_t, dtype=np.float32))
            event_adc.append(np.array(all_a, dtype=np.float32))
            event_offsets.append(np.array(pmt_offsets, dtype=np.int64))

        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=out_dir)
        os.close(fd)
        payload = {
            "time": np.array(event_time, dtype=object),
            "adc": np.array(event_adc, dtype=object),
            "offsets": np.array(event_offsets, dtype=object),
            "n_events": int(n_events),
            "file_id": int(file_id),
            "n_pmt": int(N_PMT),
            "empty_frac": float(empty_count / max(1, n_events * N_PMT)),
        }
        np.save(tmp, payload, allow_pickle=True)
        os.replace(tmp, out_path)
        return "ok", out_path
    except Exception as exc:
        err_path = os.path.join(out_dir, "errors.jsonl")
        with open(err_path, "a") as f:
            f.write(json.dumps({"file_id": file_id, "error": str(exc)}) + "\n")
        return "error", str(exc)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls_name",
                        choices=list(CLASS_ROOTS.keys()), required=True)
    parser.add_argument("--output-root", default=None,
                        help="Default: WFSampling/ or WFSampling_decon_waveform/ by kind")
    parser.add_argument("--waveform-kind", choices=["waveform", "decon_waveform"],
                        default="waveform",
                        help="Input waveform type (raw waveform inverted; decon_waveform as-is)")
    parser.add_argument("--file-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    # signal-processing parameters (match original WFSampling defaults)
    parser.add_argument("--bl-len",    type=int,   default=100)
    parser.add_argument("--thr-flat",  type=float, default=15.0)
    parser.add_argument("--global-thr",type=float, default=15.0)
    parser.add_argument("--h-thr",     type=float, default=30.0)
    parser.add_argument("--w-thr",     type=int,   default=10)
    parser.add_argument("--rise-step", type=int,   default=2)
    args = parser.parse_args()

    params = {
        "bl_len": args.bl_len, "thr_flat": args.thr_flat,
        "global_thr": args.global_thr, "h_thr": args.h_thr,
        "w_thr": args.w_thr, "rise_step": args.rise_step,
    }
    input_root = CLASS_ROOTS[args.cls_name]
    if args.output_root:
        output_root = args.output_root
    elif args.waveform_kind == "decon_waveform":
        output_root = "/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling_decon_waveform"
    else:
        output_root = OUTPUT_ROOT

    if args.file_ids:
        file_ids = args.file_ids
    else:
        wf_dir = os.path.join(input_root, "waveform")
        if args.waveform_kind == "decon_waveform":
            pattern = "decon_waveform_*.npz"
        else:
            pattern = "waveform_*.npz"
        file_ids = sorted(set(
            _file_id_from_path(p, args.waveform_kind)
            for p in glob.glob(os.path.join(wf_dir, pattern))
        ))
    if args.max_files:
        file_ids = file_ids[: args.max_files]

    out_dir = os.path.join(output_root, args.cls_name)
    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.join(out_dir, "sampling_params.json")
    with open(params_path, "w") as f:
        json.dump({"class": args.cls_name, "waveform_kind": args.waveform_kind,
                   "n_files": len(file_ids), **params}, f, indent=2)

    print(f"Processing {len(file_ids)} {args.waveform_kind} files for "
          f"class '{args.cls_name}' → {out_dir}")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_file, args.cls_name, fid, params,
                        output_root, input_root, args.waveform_kind): fid
            for fid in file_ids
        }
        for fut in as_completed(futs):
            status, msg = fut.result()
            results[status] = results.get(status, 0) + 1
            if status in ("error", "missing"):
                print(f"  [{status}] fid={futs[fut]}  {msg}")
            else:
                print(f"  [{status}] {msg}")

    print("\nSummary:", results)


if __name__ == "__main__":
    main()
