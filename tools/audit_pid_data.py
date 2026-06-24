#!/usr/bin/env python3
"""Read-only audit of numu/nue/nc PID training data."""


import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pid_lib.config import CLASS_LABEL, CLASS_ROOTS, FEATURE_FILE_PREFIX, FEATURE_NAMES, N_PMT, SUBDIRS
from pid_lib.data_io import (
    discover_file_ids,
    discover_valid_entries,
    find_waveform_files,
    inspect_npy,
    is_file_complete,
    load_aligned_file,
    load_y_arrays,
    mark_no_hit_features,
    parse_file_id,
    summarize_npz,
    FileEntry,
)


def audit_subdir(root: str, subdir: str) -> Dict:
    path = os.path.join(root, subdir)
    report = {"path": path, "exists": os.path.isdir(path), "files": []}
    if not report["exists"]:
        return report
    names = sorted(os.listdir(path))
    report["count"] = len(names)
    ids = []
    exts = defaultdict(int)
    for name in names:
        exts[os.path.splitext(name)[1]] += 1
        fid = parse_file_id(name)
        if fid is not None:
            ids.append(fid)
    report["extensions"] = dict(exts)
    if ids:
        report["id_min"] = min(ids)
        report["id_max"] = max(ids)
        report["id_count_unique"] = len(set(ids))
    # Sample first/middle/last npy or npz for schema.
    samples = []
    data_files = [n for n in names if n.endswith((".npy", ".npz"))]
    pick_idx = sorted({0, len(data_files) // 2, len(data_files) - 1})
    for idx in pick_idx:
        if 0 <= idx < len(data_files):
            fp = os.path.join(path, data_files[idx])
            try:
                if fp.endswith(".npz"):
                    samples.append(summarize_npz(fp))
                else:
                    samples.append(inspect_npy(fp))
            except Exception as exc:
                samples.append({"path": fp, "error": str(exc)})
    report["samples"] = samples
    return report


def audit_elec_fea_consistency(root: str, file_ids: List[int]) -> Dict:
    if not file_ids:
        return {"checked_files": 0}
    pick = sorted({file_ids[0], file_ids[len(file_ids) // 2], file_ids[-1]})
    issues = []
    for fid in pick:
        shapes = {}
        for feat in FEATURE_NAMES:
            p = os.path.join(root, "elec_fea", f"{FEATURE_FILE_PREFIX[feat]}_{fid}.npy")
            if os.path.isfile(p):
                arr = np.load(p, mmap_mode="r")
                shapes[feat] = list(arr.shape)
        n_events = {v[0] for v in shapes.values()}
        n_pmt = {v[1] for v in shapes.values() if len(v) >= 2}
        if len(n_events) != 1 or len(n_pmt) != 1 or list(n_pmt)[0] != N_PMT:
            issues.append({"file_id": fid, "shapes": shapes})
    return {"checked_file_ids": pick, "issues": issues}


def audit_y_alignment(entry: FileEntry) -> Dict:
    y, y_pmt = load_y_arrays(entry)
    rep = {
        "file_id": entry.file_id,
        "n_y": int(y.shape[0]),
        "n_y_pmt": int(y_pmt.shape[0]) if y_pmt is not None else None,
        "y_cols": int(y.shape[1]),
        "pid_unique": sorted({int(x) for x in np.unique(y[:, 4])}),
    }
    if y_pmt is not None:
        rep["y_pmt_cols"] = int(y_pmt.shape[1])
        rep["evt_match"] = bool(np.allclose(y[: min(y.shape[0], y_pmt.shape[0]), 1], y_pmt[: min(y.shape[0], y_pmt.shape[0]), 1]))
        rep["theta_match"] = bool(np.allclose(y[: min(y.shape[0], y_pmt.shape[0]), 2], y_pmt[: min(y.shape[0], y_pmt.shape[0]), 2]))
    return rep


def audit_class(cls_name: str, max_files_check: Optional[int] = None) -> Dict:
    root = CLASS_ROOTS[cls_name]
    rep = {
        "class": cls_name,
        "label": CLASS_LABEL[cls_name],
        "root": root,
        "root_exists": os.path.isdir(root),
        "note": None,
    }
    if cls_name == "nue" and not os.path.isdir(root):
        rep["note"] = "AGENTS.md path empty; using /disk_pool1/liuxy/nu_e/npy"
    for sub in SUBDIRS:
        rep[sub] = audit_subdir(root, sub)
    valid = [e for e in discover_valid_entries(cls_name)]
    if max_files_check is not None:
        valid = valid[:max_files_check]
    rep["valid_files"] = len(valid)
    rep["valid_events"] = 0
    rep["dropped_events"] = 0
    rep["drop_reasons"] = defaultdict(int)
    nan_inf = {"nan_events": 0, "inf_events": 0, "zero_pmt_events": 0}
    feat_ids = discover_file_ids(root, "elec_fea", FEATURE_FILE_PREFIX["fht"])
    rep["elec_fea_consistency"] = audit_elec_fea_consistency(root, feat_ids)
    y_samples = []
    for entry in valid[: min(5, len(valid))]:
        y_samples.append(audit_y_alignment(entry))
    rep["y_alignment_samples"] = y_samples
    for entry in valid:
        try:
            x, _, meta = load_aligned_file(entry)
            rep["valid_events"] += meta.get("n_kept", x.shape[0])
            rep["dropped_events"] += meta.get("dropped", 0)
            if meta.get("drop_reason"):
                rep["drop_reasons"][meta["drop_reason"]] += meta.get("dropped", 0)
            x, _ = mark_no_hit_features(x)
            if np.isnan(x).any():
                nan_inf["nan_events"] += int(np.isnan(x).reshape(x.shape[0], -1).any(axis=1).sum())
            if np.isinf(x).any():
                nan_inf["inf_events"] += int(np.isinf(x).reshape(x.shape[0], -1).any(axis=1).sum())
            zero_pmt = (np.nansum(np.abs(x), axis=(1, 2)) == 0)
            nan_inf["zero_pmt_events"] += int(zero_pmt.sum())
        except Exception as exc:
            rep.setdefault("file_errors", []).append(
                {"file_id": entry.file_id, "error": str(exc)}
            )
    rep["drop_reasons"] = dict(rep["drop_reasons"])
    rep["quality"] = nan_inf
    if valid:
        wf = find_waveform_files(root, valid[0].file_id)
        rep["waveform_example"] = {k: v for k, v in wf.items() if v}
        if wf.get("waveform"):
            rep["waveform_npz_schema"] = summarize_npz(wf["waveform"])
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit JUNO PID data (read-only).")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "audit"))
    parser.add_argument("--max-files-per-class", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "class_roots": CLASS_ROOTS,
        "class_label": CLASS_LABEL,
        "feature_names": FEATURE_NAMES,
        "classes": {},
        "total_valid_events": 0,
    }
    for cls_name in CLASS_ROOTS:
        print(f"Auditing {cls_name}...")
        crep = audit_class(cls_name, max_files_check=args.max_files_per_class)
        report["classes"][cls_name] = crep
        report["total_valid_events"] += crep.get("valid_events", 0)

    json_path = os.path.join(args.output_dir, "audit_pid_data.json")
    txt_path = os.path.join(args.output_dir, "audit_pid_data.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        f"PID Data Audit — {report['timestamp']}",
        "=" * 60,
        f"Total aligned events (numu+nue+nc): {report['total_valid_events']}",
        "",
    ]
    for cls_name, crep in report["classes"].items():
        lines.append(f"[{cls_name}] label={crep['label']} root={crep['root']}")
        lines.append(f"  valid files: {crep.get('valid_files', 0)}")
        lines.append(f"  valid events: {crep.get('valid_events', 0)}")
        lines.append(f"  dropped events: {crep.get('dropped_events', 0)}")
        if crep.get("drop_reasons"):
            lines.append(f"  drop reasons: {crep['drop_reasons']}")
        lines.append(f"  quality: {crep.get('quality', {})}")
        for sub in SUBDIRS:
            s = crep.get(sub, {})
            if s.get("exists"):
                lines.append(
                    f"  {sub}: count={s.get('count')} ids=[{s.get('id_min')},{s.get('id_max')}]"
                )
            else:
                lines.append(f"  {sub}: MISSING")
        lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Total valid events: {report['total_valid_events']}")


if __name__ == "__main__":
    main()
