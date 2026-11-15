"""Shared evaluation helpers (ROC-AUC, balanced accuracy, Youden threshold, per-magnification AUC)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y, p):
    """Threshold on P(malignant) maximising TPR - FPR (pick this on VAL, apply to TEST)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def summarize(y, p, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec_b = tn / max(tn + fp, 1)
    rec_m = tp / max(tp + fn, 1)
    return {
        "auc": safe_auc(y, p),
        "threshold": float(thr),
        "accuracy": float((tn + tp) / cm.sum()),
        "balanced_accuracy": float((rec_b + rec_m) / 2),
        "benign_recall": float(rec_b),
        "malignant_recall": float(rec_m),
        "confusion_matrix": cm.tolist(),
    }


def per_magnification_auc(y, p, mags):
    y, p, mags = np.asarray(y), np.asarray(p), np.asarray(mags)
    return {m: safe_auc(y[mags == m], p[mags == m]) for m in sorted(set(mags))}


def save_roc_plot(y, p, path, title="MASNN ROC (test)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {safe_auc(y, p):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
