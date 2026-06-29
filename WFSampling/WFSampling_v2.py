#!/usr/bin/env python3
"""WFSampling v2 — Python port of WFSampling_v2.cc key-point extraction.

Differences from v1 (WFSampling/v1/WFSampling_v1.py):
  - FHT gate: pulses only counted after AC exceeds thr_fht (fraction or absolute).
  - First peak: rise_step sampling + 55% peak amplitude point.
  - Improved end-of-waveform pulse closure.

Usage:
  python3 WFSampling/WFSampling_v2.py --class numu --workers 4
  python3 WFSampling/WFSampling_v2.py --class nue --waveform-kind decon_waveform
"""

import argparse
import datetime
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_ROOTS, N_PMT

OUTPUT_ROOT_WAV_V2 = "/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2"
OUTPUT_ROOT_DECON_WAV_V2 = "/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2"


def default_output_root(waveform_kind):
    if waveform_kind == "decon_waveform":
        return OUTPUT_ROOT_DECON_WAV_V2
    return OUTPUT_ROOT_WAV_V2


def default_global_thr(waveform_kind):
    return 1.0 if waveform_kind == "decon_waveform" else 15.0


def default_h_thr(waveform_kind):
    """Height threshold for key-point confirmation.

    decon_waveform AC values are scaled by /100 relative to raw ADC, so the
    threshold must be proportionally smaller (0.5 vs 50).
    """
    return 0.5 if waveform_kind == "decon_waveform" else 50.0


def get_baseline_and_ac(adc, bl_len=100, thr_flat=15.0):
    adc = adc.astype(np.float32)
    piece = 8
    flag_good = False
    start = 0
    for start in range(0, bl_len - 3 * piece, piece):
        seg = adc[start:start + piece]
        bl = np.mean(seg)
        if np.all(np.abs(seg - bl) <= thr_flat):
            flag_good = True
            break
    if not flag_good:
        baseline_val = float(np.mean(adc[:bl_len]))
    else:
        bl = float(np.mean(adc[start:start + piece]))
        end = bl_len
        for i in range(start, bl_len):
            if abs(adc[i] - bl) > thr_flat:
                end = i
                break
        baseline_val = float(np.mean(adc[start:end]))
    ac = adc - baseline_val
    ac = np.clip(ac, 0, None)
    return ac, baseline_val


def extract_key_points_v2(
    ac,
    global_thr=15.0,
    h_thr=30.0,
    w_thr=10,
    rise_step=2,
    thr_fht=0.55,
):
    """Port of WFSampling::extractKeyPoints in WFSampling_v2.cc."""
    ac = np.asarray(ac, dtype=np.float32)
    n = len(ac)
    if n == 0:
        return np.array([], np.float32), np.array([], np.float32)

    out_t, out_a = [], []
    in_pulse = False
    looking_for_peak = False
    last_valley_time = 0
    last_valley_adc = 0.0
    local_max_adc = -np.inf
    local_max_time = -1
    local_min_adc = np.inf
    local_min_time = -1
    confirmed_peak_time = -1
    confirmed_peak_adc = -1.0
    first_peak_sampled = False

    peak_value = float(np.max(ac))
    if thr_fht >= 1.0:
        threshold_fht = float(thr_fht)
    else:
        threshold_fht = float(thr_fht) * peak_value
    thr_slope = peak_value * 0.55
    found_fht = False

    def add_point(t, val):
        if not out_t or float(t) > out_t[-1]:
            out_t.append(float(t))
            out_a.append(float(val))

    def sample_first_peak(start_t, end_t):
        nonlocal first_peak_sampled
        temp = []
        for t in range(int(start_t) + rise_step, int(end_t), rise_step):
            if 0 <= t < n:
                temp.append((t, float(ac[t])))
        idx_55 = int(start_t)
        best_diff = abs(float(ac[idx_55]) - thr_slope)
        for t in range(int(start_t) + 1, int(end_t) + 1):
            if t < 0 or t >= n:
                continue
            diff = abs(float(ac[t]) - thr_slope)
            if diff < best_diff:
                best_diff = diff
                idx_55 = t
        temp.append((idx_55, float(ac[idx_55])))
        temp.sort(key=lambda x: x[0])
        dedup = []
        for pt in temp:
            if not dedup or pt[0] != dedup[-1][0]:
                dedup.append(pt)
        for t, v in dedup:
            add_point(t, v)
        first_peak_sampled = True

    for i in range(1, n - 1):
        curr = float(ac[i])
        if curr > threshold_fht and not found_fht:
            found_fht = True

        if curr > global_thr and found_fht:
            if not in_pulse:
                in_pulse = True
                looking_for_peak = True
                last_valley_time = i
                last_valley_adc = curr
                local_max_adc = curr
                local_max_time = i
                local_min_adc = curr
                local_min_time = i

            if curr > local_max_adc:
                local_max_adc = curr
                local_max_time = i
            if curr < local_min_adc:
                local_min_adc = curr
                local_min_time = i

            if looking_for_peak:
                if curr < local_max_adc - h_thr:
                    confirmed_peak_time = local_max_time
                    confirmed_peak_adc = local_max_adc
                    looking_for_peak = False
                    local_min_adc = curr
                    local_min_time = i
            else:
                if curr > local_min_adc + h_thr:
                    width = local_min_time - last_valley_time
                    if width >= w_thr:
                        add_point(last_valley_time, last_valley_adc)
                        sample_first_peak(last_valley_time, confirmed_peak_time)
                        add_point(confirmed_peak_time, confirmed_peak_adc)
                        add_point(local_min_time, local_min_adc)
                        last_valley_time = local_min_time
                        last_valley_adc = local_min_adc
                        local_max_adc = curr
                        local_max_time = i
                    looking_for_peak = True
        else:
            if in_pulse:
                in_pulse = False
                if not looking_for_peak:
                    width = i - last_valley_time
                    if width >= w_thr:
                        add_point(last_valley_time, last_valley_adc)
                        if not first_peak_sampled:
                            sample_first_peak(last_valley_time, confirmed_peak_time)
                        add_point(confirmed_peak_time, confirmed_peak_adc)
                        add_point(i, curr)
                else:
                    if local_max_adc - last_valley_adc >= h_thr:
                        width = i - last_valley_time
                        if width >= w_thr:
                            add_point(last_valley_time, last_valley_adc)
                            if not first_peak_sampled:
                                sample_first_peak(last_valley_time, confirmed_peak_time)
                            add_point(local_max_time, local_max_adc)
                            add_point(i, curr)

    if in_pulse:
        last_idx = n - 2
        last_curr = float(ac[last_idx])
        if not looking_for_peak:
            width = last_idx - last_valley_time
            if width >= w_thr:
                add_point(last_valley_time, last_valley_adc)
                sample_first_peak(last_valley_time, confirmed_peak_time)
                add_point(confirmed_peak_time, confirmed_peak_adc)
                add_point(last_idx, last_curr)
        else:
            if local_max_adc - last_valley_adc >= h_thr:
                width = last_idx - last_valley_time
                if width >= w_thr:
                    add_point(last_valley_time, last_valley_adc)
                    if not first_peak_sampled:
                        sample_first_peak(last_valley_time, confirmed_peak_time)
                    add_point(local_max_time, local_max_adc)
                    add_point(last_idx, last_curr)

    return np.array(out_t, np.float32), np.array(out_a, np.float32)


def _load_waveform_events(wf_path, waveform_kind="waveform"):
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


def process_file(cls_name, file_id, params, output_root, input_root, waveform_kind="waveform", force=False):
    out_dir = os.path.join(output_root, cls_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wfsampling_{file_id}.npy")

    if not force and os.path.isfile(out_path):
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
        event_time, event_adc, event_offsets = [], [], []
        empty_count = 0
        invert = waveform_kind == "waveform"
        for dense in events:
            all_t, all_a = [], []
            pmt_offsets = [0]
            for pmt in range(N_PMT):
                raw = dense[pmt]
                signal = (16384.0 - raw) if invert else raw
                if waveform_kind == "decon_waveform":
                    ac = (signal - 1000.0) / 100.0
                    ac = np.clip(ac, 0, None)
                else:
                    ac, _ = get_baseline_and_ac(
                        signal, bl_len=params["bl_len"], thr_flat=params["thr_flat"],
                    )
                t_pts, a_pts = extract_key_points_v2(
                    ac,
                    global_thr=params["global_thr"],
                    h_thr=params["h_thr"],
                    w_thr=params["w_thr"],
                    rise_step=params["rise_step"],
                    thr_fht=params["thr_fht"],
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
            "time": [t.tolist() for t in event_time],
            "adc": [a.tolist() for a in event_adc],
            "offsets": [o.tolist() for o in event_offsets],
            "n_events": int(len(events)),
            "file_id": int(file_id),
            "n_pmt": int(N_PMT),
            "empty_frac": float(empty_count / max(1, len(events) * N_PMT)),
            "wfs_version": 2,
        }
        np.save(tmp, payload, allow_pickle=True)
        os.replace(tmp, out_path)
        return "ok", out_path
    except Exception as exc:
        err_path = os.path.join(out_dir, "errors.jsonl")
        with open(err_path, "a") as f:
            f.write(json.dumps({"file_id": file_id, "error": str(exc)}) + "\n")
        return "error", str(exc)


def main():
    parser = argparse.ArgumentParser(description="WFSampling v2 (port of WFSampling_v2.cc)")
    parser.add_argument("--class", dest="cls_name", choices=list(CLASS_ROOTS.keys()), required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--waveform-kind", choices=["waveform", "decon_waveform"], default="waveform")
    parser.add_argument("--file-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--bl-len", type=int, default=100)
    parser.add_argument("--thr-flat", type=float, default=15.0)
    parser.add_argument("--global-thr", type=float, default=None,
                        help="Pulse threshold; default 15 (waveform) or 1 (decon_waveform)")
    parser.add_argument("--thr-fht", type=float, default=0.2,
                        help="FHT gate: fraction of peak if <1, else absolute ADC")
    parser.add_argument("--h-thr", type=float, default=None,
                        help="Peak height threshold; default 50 (waveform) or 0.5 (decon_waveform)")
    parser.add_argument("--w-thr", type=int, default=15)
    parser.add_argument("--rise-step", type=int, default=2)
    args = parser.parse_args()

    global_thr = args.global_thr if args.global_thr is not None else default_global_thr(args.waveform_kind)
    h_thr = args.h_thr if args.h_thr is not None else default_h_thr(args.waveform_kind)
    params = {
        "bl_len": args.bl_len,
        "thr_flat": args.thr_flat,
        "global_thr": global_thr,
        "thr_fht": args.thr_fht,
        "h_thr": h_thr,
        "w_thr": args.w_thr,
        "rise_step": args.rise_step,
        "wfs_version": 2,
        "waveform_kind": args.waveform_kind,
    }
    input_root = CLASS_ROOTS[args.cls_name]
    output_root = args.output_root or default_output_root(args.waveform_kind)

    if args.file_ids:
        file_ids = args.file_ids
    else:
        wf_dir = os.path.join(input_root, "waveform")
        pattern = "decon_waveform_*.npz" if args.waveform_kind == "decon_waveform" else "waveform_*.npz"
        file_ids = sorted({
            _file_id_from_path(p, args.waveform_kind)
            for p in glob.glob(os.path.join(wf_dir, pattern))
        })
    if args.max_files:
        file_ids = file_ids[: args.max_files]

    out_dir = os.path.join(output_root, args.cls_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[WFSampling_v2] {len(file_ids)} {args.waveform_kind} files "
          f"class={args.cls_name} h_thr={h_thr} global_thr={global_thr} -> {out_dir}")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_file, args.cls_name, fid, params, output_root,
                        input_root, args.waveform_kind, args.force): fid
            for fid in file_ids
        }
        for fut in as_completed(futs):
            status, msg = fut.result()
            results[status] = results.get(status, 0) + 1
            if status in ("error", "missing"):
                print(f"  [{status}] fid={futs[fut]}  {msg}")

    summary = {
        "class": args.cls_name,
        "waveform_kind": args.waveform_kind,
        "n_files": len(file_ids),
        **params,
        "results": results,
        "n_files_processed": results.get("ok", 0) + results.get("skipped", 0),
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "sampling_params.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary:", results)


if __name__ == "__main__":
    main()
