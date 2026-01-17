import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

def parse_args():
    p = argparse.ArgumentParser("plots (PID 3-class)")
    p.add_argument("--log_dir", type=str, default="/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test1/plots")
    p.add_argument("--pred_file", type=str, default="/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test1/predictions_test.npz")
    return p.parse_args()

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def plot_confusion(cm: np.ndarray, labels, out_path: str, title: str, normalize: bool = False, values_format: str = None):
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    if normalize:
        disp.plot(ax=ax, cmap="Blues", values_format=values_format or ".3f", colorbar=True)
    else:
        disp.plot(ax=ax, cmap="Blues", values_format=values_format or "d", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def _confusion_efficiency(cm: np.ndarray) -> np.ndarray:
    """
    Efficiency-style normalization (row-normalized):
      eff[i,j] = P(pred=j | true=i)
    Rows=true class, Cols=pred class.
    """
    cm = cm.astype(np.float64, copy=False)
    row_sum = cm.sum(axis=1, keepdims=True)
    eff = cm / np.maximum(row_sum, 1.0)
    return eff.astype(np.float32)

def plot_ovr_roc_pr(y_true: np.ndarray, prob: np.ndarray, class_names, out_dir: str):
    """
    One-vs-rest ROC & PR for each class.
    """
    C = prob.shape[1]
    for c in range(C):
        y_bin = (y_true == c).astype(np.int64)
        p = prob[:, c].astype(np.float32)

        # ROC
        fpr, tpr, _ = roc_curve(y_bin, p)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.5f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC OvR (positive={class_names[c]})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"roc_ovr_{class_names[c]}.png"), dpi=170)
        plt.close(fig)

        # PR
        prec, rec, _ = precision_recall_curve(y_bin, p)
        ap = average_precision_score(y_bin, p)
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        ax.plot(rec, prec, label=f"AP={ap:.5f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR OvR (positive={class_names[c]})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pr_ovr_{class_names[c]}.png"), dpi=170)
        plt.close(fig)

def plot_eff_roc_fig9_style(y_true: np.ndarray, prob: np.ndarray, class_names, out_path: str):
    """
    FIG.9 style efficiency ROC curves (OvR with paper axis labeling).
    For class k:
      y-axis: efficiency selecting class k (TPR)
      x-axis: efficiency selecting the other two combined (FPR)
    """
    C = len(class_names)
    fig, axes = plt.subplots(1, C, figsize=(5.4 * C, 4.6), squeeze=False)
    axes = axes[0]

    for k in range(C):
        y_bin = (y_true == k).astype(np.int32)
        score = prob[:, k].astype(np.float32)

        fpr, tpr, _ = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)

        ax = axes[k]
        ax.plot(fpr, tpr, color="red", linewidth=2, label=f"AUC = {roc_auc:.2f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.35)

        other = [class_names[j] for j in range(C) if j != k]
        other_str = "/".join(other)
        ax.set_xlabel(f"{other_str} Efficiency")
        ax.set_ylabel(f"{class_names[k]} Efficiency")
        ax.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_eff_roc_pair_fig9(y_true: np.ndarray, prob: np.ndarray, class_names, out_path: str):
    """
    2-panel variant similar to pasted figure:
      - Vmu vs (Ve + nc)
      - Ve  vs (Vmu + nc)
    Requires class_names containing 'nc','Vmu','Ve'.
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    required = ["nc", "Vmu", "Ve"]
    if not all(n in name_to_idx for n in required):
        # if names different, skip silently to avoid breaking
        return

    i_mu = name_to_idx["Vmu"]
    i_e = name_to_idx["Ve"]

    setups = [
        ("Vmu", i_mu, "Ve/nc + NC Efficiency", "Vmu Efficiency"),
        ("Ve",  i_e,  "Vmu/nc + NC Efficiency", "Ve Efficiency"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for ax, (pos_name, pos_idx, xlabel, ylabel) in zip(axes, setups):
        y_bin = (y_true == pos_idx).astype(np.int32)
        score = prob[:, pos_idx].astype(np.float32)
        fpr, tpr, _ = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color="red", linewidth=2, label=f"AUC = {roc_auc:.2f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main(args):
    out_dir = ensure_dir(args.log_dir)

    d = np.load(args.pred_file, allow_pickle=True)
    if "pred_class" not in d.files or "true_class" not in d.files:
        raise KeyError(f"pred_file missing keys. got keys={list(d.files)} from {args.pred_file}")

    pred_class = d["pred_class"].astype(np.int64)
    true_class = d["true_class"].astype(np.int64)
    class_names = d["class_names"].tolist() if "class_names" in d.files else ["nc", "Vmu", "Ve"]
    prob = d["prob"].astype(np.float32) if "prob" in d.files else None
    print("[PLOTS] true_class bincount:", np.bincount(true_class, minlength=len(class_names)).tolist())

    # metrics
    acc = float(accuracy_score(true_class, pred_class))
    prec, rec, f1, support = precision_recall_fscore_support(
        true_class, pred_class, labels=list(range(len(class_names))), zero_division=0
    )

    metrics = {
        "num_samples": int(true_class.size),
        "accuracy": acc,
        "class_names": class_names,
        "per_class": {
            class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # confusion matrices
    cm = confusion_matrix(true_class, pred_class, labels=list(range(len(class_names))))
    plot_confusion(
        cm,
        labels=class_names,
        out_path=os.path.join(out_dir, "confusion_counts.png"),
        title="Confusion Matrix (Counts)",
        normalize=False,
    )

    # efficiency-normalized confusion matrix (P(pred|true))
    cm_eff = _confusion_efficiency(cm)
    plot_confusion(
        cm_eff,
        labels=class_names,
        out_path=os.path.join(out_dir, "confusion_efficiency.png"),
        title="Confusion Matrix (Efficiency: P(pred|true))",
        normalize=True,
        values_format=".3f",
    )

    # curves (always plot if prob exists)
    if prob is not None:
        if prob.ndim != 2 or prob.shape[1] != len(class_names):
            raise ValueError(f"Bad prob shape: {prob.shape}, expected [N, {len(class_names)}]")

        plot_ovr_roc_pr(true_class, prob, class_names, out_dir=out_dir)
        plot_eff_roc_fig9_style(
            true_class, prob, class_names, out_path=os.path.join(out_dir, "roc_efficiency_fig9_3panels.png")
        )
        plot_eff_roc_pair_fig9(
            true_class, prob, class_names, out_path=os.path.join(out_dir, "roc_efficiency_fig9_2panels.png")
        )
    else:
        print("[WARN] prob not found in npz, skipping ROC/PR & efficiency ROC plots. Re-run evaluation.py to save prob.")

    print(f"[OK] saved plots to: {out_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(parse_args())