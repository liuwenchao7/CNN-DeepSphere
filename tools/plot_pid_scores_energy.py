#!/usr/bin/env python3
"""EXECUTION_PLAN Task 2 plots: score distributions and visE-binned AUC.

Implements paper-like plotting style and bootstrap uncertainty described in
docs/SCORE_AND_BOOTSTRAP.md.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from pid_lib.splits import load_manifest
from pid_lib.data_io import load_y_arrays, align_event_indices


CLASS_NAMES = ["numu", "nue", "nc"]
COLORS = {"numu": "#FF4C4C", "nue": "#4CA64C", "nc": "#4C4CFF"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Model output dir; repeat this flag for multiple models.",
    )
    p.add_argument(
        "--model-label",
        action="append",
        default=[],
        help="Display label for each --model-dir; same count/order as model-dir.",
    )
    p.add_argument("--output-dir", default=os.path.join(ROOT, "audit", "task7_plots"))
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--energy-min", type=float, default=0.5)
    p.add_argument("--energy-max", type=float, default=15.0)
    p.add_argument("--energy-bins", type=int, default=12)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    return p.parse_args()


def load_vise_from_manifest(model_dir):
    manifest = os.path.join(model_dir, "manifest_test.json")
    if not os.path.isfile(manifest):
        return None
    entries = load_manifest(manifest)
    all_vise = []
    for e in entries:
        y, y_pmt = load_y_arrays(e)
        if y_pmt is not None and y_pmt.shape[0] != y.shape[0]:
            idx = align_event_indices(y, y_pmt)
            y = y[idx]
        # Columns used across this project: [*, evt, theta, phi, pid, visE, ...]
        if y.shape[1] <= 5:
            return None
        all_vise.append(y[:, 5].astype(np.float32))
    if not all_vise:
        return None
    return np.concatenate(all_vise)


def ovr_auc(y_true, y_prob, cls_idx):
    y_bin = (y_true == cls_idx).astype(np.int32)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls_idx])
    return float(auc(fpr, tpr))


def stratified_bootstrap_indices(y_true, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    cls_idx = {c: np.where(y_true == c)[0] for c in range(3)}
    out = []
    for _ in range(n_boot):
        picks = []
        for c in range(3):
            idx = cls_idx[c]
            if len(idx) == 0:
                continue
            picks.append(rng.choice(idx, size=len(idx), replace=True))
        if not picks:
            out.append(np.array([], dtype=np.int64))
            continue
        out.append(np.concatenate(picks))
    return out


def bootstrap_auc_stats(y_true, y_prob, cls_idx, boot_indices):
    vals = []
    for idx in boot_indices:
        if len(idx) == 0:
            continue
        a = ovr_auc(y_true[idx], y_prob[idx], cls_idx)
        if not np.isnan(a):
            vals.append(a)
    if len(vals) < 5:
        return np.nan, np.nan, np.nan, np.nan
    vals = np.array(vals, dtype=np.float32)
    return (
        float(np.std(vals, ddof=1)),
        float(np.nanpercentile(vals, 16)),
        float(np.nanpercentile(vals, 50)),
        float(np.nanpercentile(vals, 84)),
    )


def _paper_axes_style(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", direction="in", length=6, width=1.0)
    ax.grid(False)


def plot_score_hists(y_true, y_prob, model_label, output_dir, bins):
    os.makedirs(output_dir, exist_ok=True)
    score_map = {"mu_like": 0, "e_like": 1}
    panel_data = {}
    for score_name, score_idx in score_map.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            v = y_prob[y_true == cls_idx, score_idx]
            if len(v) == 0:
                continue
            ax.hist(
                v,
                bins=bins,
                range=(0.0, 1.0),
                density=False,
                histtype="step",
                linewidth=1.5,
                color=COLORS[cls_name],
                label=f"true={cls_name} (n={len(v)})",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Model score")
        ax.set_ylabel("Arbitrary Scale")
        ax.set_yscale("log")
        ax.set_title(f"{model_label}: {score_name} score distribution")
        _paper_axes_style(ax)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(output_dir, f"score_{score_name}.png")
        plt.savefig(out, dpi=200)
        plt.savefig(os.path.join(output_dir, f"score_{score_name}.pdf"))
        panel_data[score_name] = (y_true.copy(), y_prob[:, score_idx].copy())
        plt.close()
    return panel_data


def plot_energy_auc(y_true, y_prob, vise, model_label, output_dir, e_edges, n_boot=1000, boot_seed=42):
    rows = []
    for i in range(len(e_edges) - 1):
        lo, hi = e_edges[i], e_edges[i + 1]
        if i == len(e_edges) - 2:
            sel = (vise >= lo) & (vise <= hi)
        else:
            sel = (vise >= lo) & (vise < hi)
        yt = y_true[sel]
        yp = y_prob[sel]
        row = {
            "E_low": float(lo),
            "E_high": float(hi),
            "E_center": float(0.5 * (lo + hi)),
            "E_half_width": float(0.5 * (hi - lo)),
            "n_total": int(np.sum(sel)),
            "n_numu": int(np.sum(yt == 0)),
            "n_nue": int(np.sum(yt == 1)),
            "n_nc": int(np.sum(yt == 2)),
        }
        aucs = [ovr_auc(yt, yp, 0), ovr_auc(yt, yp, 1), ovr_auc(yt, yp, 2)] if len(yt) else [np.nan, np.nan, np.nan]
        boot_idx = stratified_bootstrap_indices(yt, n_boot=n_boot, seed=boot_seed + i) if len(yt) else []
        bstats = [bootstrap_auc_stats(yt, yp, 0, boot_idx), bootstrap_auc_stats(yt, yp, 1, boot_idx), bootstrap_auc_stats(yt, yp, 2, boot_idx)] if len(yt) else [(np.nan, np.nan, np.nan, np.nan)] * 3
        row["auc_numu"] = float(aucs[0]) if not np.isnan(aucs[0]) else None
        row["auc_nue"] = float(aucs[1]) if not np.isnan(aucs[1]) else None
        row["auc_nc"] = float(aucs[2]) if not np.isnan(aucs[2]) else None
        row["auc_numu_boot_sigma"] = float(bstats[0][0]) if not np.isnan(bstats[0][0]) else None
        row["auc_nue_boot_sigma"] = float(bstats[1][0]) if not np.isnan(bstats[1][0]) else None
        row["auc_nc_boot_sigma"] = float(bstats[2][0]) if not np.isnan(bstats[2][0]) else None
        row["auc_numu_boot_q16"] = float(bstats[0][1]) if not np.isnan(bstats[0][1]) else None
        row["auc_numu_boot_q50"] = float(bstats[0][2]) if not np.isnan(bstats[0][2]) else None
        row["auc_numu_boot_q84"] = float(bstats[0][3]) if not np.isnan(bstats[0][3]) else None
        row["auc_nue_boot_q16"] = float(bstats[1][1]) if not np.isnan(bstats[1][1]) else None
        row["auc_nue_boot_q50"] = float(bstats[1][2]) if not np.isnan(bstats[1][2]) else None
        row["auc_nue_boot_q84"] = float(bstats[1][3]) if not np.isnan(bstats[1][3]) else None
        row["auc_nc_boot_q16"] = float(bstats[2][1]) if not np.isnan(bstats[2][1]) else None
        row["auc_nc_boot_q50"] = float(bstats[2][2]) if not np.isnan(bstats[2][2]) else None
        row["auc_nc_boot_q84"] = float(bstats[2][3]) if not np.isnan(bstats[2][3]) else None
        valid = [a for a in aucs if not np.isnan(a)]
        row["auc_macro"] = float(np.mean(valid)) if valid else None
        macro_sig = [x[0] for x in bstats if not np.isnan(x[0])]
        row["auc_macro_boot_sigma"] = float(np.mean(macro_sig)) if macro_sig else None
        rows.append(row)

    with open(os.path.join(output_dir, "energy_auc_bins.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    x = np.array([(r["E_low"] + r["E_high"]) / 2.0 for r in rows], dtype=np.float32)
    def arr(key):
        out = []
        for r in rows:
            v = r[key]
            out.append(np.nan if v is None else v)
        return np.array(out, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 4))
    xerr = np.array([r["E_half_width"] for r in rows], dtype=np.float32)
    ax.errorbar(x, arr("auc_macro"), yerr=arr("auc_macro_boot_sigma"), xerr=xerr,
                fmt="o", linestyle="none", capsize=3, label="macro")
    ax.errorbar(x, arr("auc_numu"), yerr=arr("auc_numu_boot_sigma"), xerr=xerr,
                fmt="o", linestyle="none", capsize=3, label="numu")
    ax.errorbar(x, arr("auc_nue"), yerr=arr("auc_nue_boot_sigma"), xerr=xerr,
                fmt="o", linestyle="none", capsize=3, label="nue")
    ax.errorbar(x, arr("auc_nc"), yerr=arr("auc_nc_boot_sigma"), xerr=xerr,
                fmt="o", linestyle="none", capsize=3, label="nc")
    ax.set_xlabel("Visible energy (visE)")
    ax.set_ylabel("One-vs-rest AUC")
    ax.set_ylim(0.8, 1.0)
    ax.set_title(f"{model_label}: AUC vs visE")
    ax.grid(True, alpha=0.25, color="#DDDDDD")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auc_vs_vise.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "auc_vs_vise.pdf"))
    plt.close()
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = args.model_dir
    if not model_dirs:
        model_dirs = [
            os.path.join(ROOT, "outputs", "fea6"),
            os.path.join(ROOT, "outputs", "fea6+decon_npevst"),
            os.path.join(ROOT, "outputs", "fea6+decon_waveform"),
            os.path.join(ROOT, "outputs", "fea6+wfsampling"),
        ]
    labels = args.model_label if args.model_label else [os.path.basename(m) for m in model_dirs]
    if len(labels) != len(model_dirs):
        raise ValueError("--model-label count must equal --model-dir count")

    summary = []
    e_edges = np.linspace(args.energy_min, args.energy_max, args.energy_bins + 1, dtype=np.float32)
    comp_mu = []
    comp_e = []
    comp_auc = []
    score_panels = []

    for model_dir, model_label in zip(model_dirs, labels):
        y_true_path = os.path.join(model_dir, "y_true.npy")
        y_prob_path = os.path.join(model_dir, "y_prob.npy")
        vise_path = os.path.join(model_dir, "vise_test.npy")
        if not (os.path.isfile(y_true_path) and os.path.isfile(y_prob_path)):
            summary.append({"model": model_label, "dir": model_dir, "status": "missing y_true/y_prob"})
            continue

        y_true = np.load(y_true_path).astype(np.int32)
        y_prob = np.load(y_prob_path).astype(np.float32)
        out_dir = os.path.join(args.output_dir, model_label.replace("/", "_"))
        os.makedirs(out_dir, exist_ok=True)

        score_panels.append((model_label, plot_score_hists(y_true, y_prob, model_label, out_dir, args.bins)))

        vise = None
        if os.path.isfile(vise_path):
            vise = np.load(vise_path).astype(np.float32)
        else:
            vise = load_vise_from_manifest(model_dir)
        if vise is not None and len(vise) != len(y_true):
            print(f"[warn] {model_label}: visE len {len(vise)} != y_true {len(y_true)}; skip energy AUC")
            vise = None
            rows = plot_energy_auc(y_true, y_prob, vise, model_label, out_dir, e_edges,
                                   n_boot=args.bootstrap, boot_seed=args.bootstrap_seed)
            comp_auc.append((model_label, rows))
            status = "ok"
        else:
            status = "score-only (visE unavailable or size mismatch)"

        # for global comparison figures
        comp_mu.append((model_label, y_true, y_prob[:, 0]))
        comp_e.append((model_label, y_true, y_prob[:, 1]))
        summary.append({"model": model_label, "dir": model_dir, "status": status, "n_test": int(len(y_true))})

    # Unified score comparison figures
    def plot_comp_score(comp_data, score_name, out_path):
        n = max(1, len(comp_data))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for i, (label, y_true, score) in enumerate(comp_data):
            ax = axes[0, i]
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                v = score[y_true == cls_idx]
                if len(v) == 0:
                    continue
                ax.hist(v, bins=args.bins, range=(0, 1), density=False,
                        histtype="step", linewidth=1.5,
                        color=COLORS[cls_name], label=f"{cls_name} (n={len(v)})")
            ax.set_xlim(0, 1)
            ax.set_yscale("log")
            ax.set_title(label)
            ax.set_xlabel(f"{score_name} score")
            ax.set_ylabel("Arbitrary Scale")
            _paper_axes_style(ax)
            if i == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.savefig(out_path.replace(".png", ".pdf"))
        plt.close()

    if comp_mu:
        plot_comp_score(comp_mu, "mu-like", os.path.join(args.output_dir, "compare_mu_like_scores.png"))
    if comp_e:
        plot_comp_score(comp_e, "e-like", os.path.join(args.output_dir, "compare_e_like_scores.png"))

    # Unified macro AUC-vs-E comparison
    if comp_auc:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, rows in comp_auc:
            x = np.array([(r["E_low"] + r["E_high"]) / 2.0 for r in rows], dtype=np.float32)
            y = np.array([np.nan if r["auc_macro"] is None else r["auc_macro"] for r in rows], dtype=np.float32)
            yerr = np.array([np.nan if r.get("auc_macro_boot_sigma") is None else r.get("auc_macro_boot_sigma") for r in rows], dtype=np.float32)
            xerr = np.array([r["E_half_width"] for r in rows], dtype=np.float32)
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o",
                        linestyle="none", capsize=3, label=label)
        ax.set_xlabel("Visible energy (visE)")
        ax.set_ylabel("Macro one-vs-rest AUC")
        ax.set_ylim(0.8, 1.0)
        ax.grid(True, alpha=0.25, color="#DDDDDD")
        ax.legend()
        ax.set_title("Macro AUC vs visE (model comparison)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "compare_macro_auc_vs_vise.png"), dpi=200)
        plt.savefig(os.path.join(args.output_dir, "compare_macro_auc_vs_vise.pdf"))
        plt.close()

    # 4x2 overview (or Nx2 for current number of models)
    if score_panels:
        n = len(score_panels)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
        for i, (label, pdata) in enumerate(score_panels):
            for j, (score_name, score_idx) in enumerate([("mu_like", 0), ("e_like", 1)]):
                ax = axes[i, j]
                y_true, score = pdata[score_name]
                for cls_idx, cls_name in enumerate(CLASS_NAMES):
                    v = score[y_true == cls_idx]
                    if len(v) == 0:
                        continue
                    ax.hist(v, bins=args.bins, range=(0, 1),
                            histtype="step", linewidth=1.5,
                            color=COLORS[cls_name], label=cls_name if (i == 0 and j == 0) else None)
                ax.set_xlim(0, 1)
                ax.set_yscale("log")
                ax.set_ylabel("Arbitrary Scale")
                ax.set_xlabel(f"{score_name} score")
                ax.set_title(label if j == 0 else "")
                _paper_axes_style(ax)
        if n > 0:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "score_overview_4x2.png"), dpi=220)
        plt.savefig(os.path.join(args.output_dir, "score_overview_4x2.pdf"))
        plt.close()

    style_meta = {
        "score_colors": COLORS,
        "score_histtype": "step",
        "score_y_scale": "log",
        "bootstrap": {
            "n_boot": args.bootstrap,
            "seed": args.bootstrap_seed,
            "sampling": "stratified_with_replacement_per_class",
            "errorbar_vertical": "bootstrap std (ddof=1)",
            "errorbar_horizontal": "energy bin half width",
        },
    }
    with open(os.path.join(args.output_dir, "plot_style_and_bootstrap.json"), "w", encoding="utf-8") as f:
        json.dump(style_meta, f, indent=2)

    with open(os.path.join(args.output_dir, "plot_task7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
