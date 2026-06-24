"""Classification metrics and plots for 3-class PID."""


import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)

from pid_lib.config import CLASS_LABEL, LABEL_TO_CLASS


def compute_class_weights(y_true: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    n_cls = len(classes)
    weights = {}
    for c, cnt in zip(classes, counts):
        weights[int(c)] = float(total / (n_cls * cnt))
    return weights


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    label_names: Optional[Sequence[str]] = None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    if label_names is None:
        label_names = [LABEL_TO_CLASS[i] for i in sorted(CLASS_LABEL.values())]

    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    roc_data = {}
    aucs = []
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(label_names):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-rest ROC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_ovr.png"), dpi=200)
    plt.close()

    cm_row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_ratio = cm.astype(np.float32) / cm_row_sum.astype(np.float32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_ratio": cm_ratio.tolist(),
        "per_class": {
            label_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "macro_auc": float(np.mean(aucs)),
        "per_class_auc": {label_names[i]: float(aucs[i]) for i in range(3)},
    }

    np.save(os.path.join(output_dir, "y_true.npy"), y_true)
    np.save(os.path.join(output_dir, "y_prob.npy"), y_prob)
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_ratio, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_ratio[i, j]:.3f}", ha="center", va="center")
    plt.colorbar(im)
    plt.title("Confusion Matrix (row-normalized ratio)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def save_class_mapping(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_LABEL, f, indent=2)
