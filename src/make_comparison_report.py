"""
Generates presentation-ready comparison figures: the original broken baseline run
vs. the fixed run, side by side. Saves standalone PNG files to data/splits/.

Run: venv\\Scripts\\python.exe src\\make_comparison_report.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# --- Original (broken) run: recorded from the very first training execution,
# before any of the fixes (lr=1e-3, no class weighting, no scheduler/checkpointing). ---
OLD_HISTORY = {
    "train_loss": [0.3875, 0.3269, 0.3194, 0.2986, 0.2919, 0.2832, 0.2808, 0.2655],
    "val_loss":   [0.4119, 0.3156, 0.5104, 0.2739, 0.6216, 1.1203, 1.4732, 0.3606],
    "train_acc":  [0.825, 0.852, 0.858, 0.867, 0.871, 0.877, 0.876, 0.883],
    "val_acc":    [0.838, 0.882, 0.778, 0.888, 0.738, 0.527, 0.461, 0.885],
}
OLD_REPORT = {
    "benign": {"precision": 0.39, "recall": 0.22, "f1-score": 0.28, "support": 250},
    "malignant": {"precision": 0.83, "recall": 0.92, "f1-score": 0.87, "support": 1032},
    "accuracy": 0.78,
}
OLD_CM = np.array([[55, 195], [82, 950]])  # reconstructed from 22% benign recall / 92% malignant recall on 250/1032 support (raw counts not saved from that run, only the printed report)

# --- Fixed baseline CNN (Week 2, v2: augmentation + accuracy-based checkpointing) ---
with open("data/splits/week2_results.json") as f:
    NEW = json.load(f)
NEW_HISTORY = NEW["history"]
NEW_REPORT = NEW["classification_report"]
NEW_CM = np.array(NEW["confusion_matrix"])

# --- Spiking network (Week 3) ---
with open("data/splits/week3_results.json") as f:
    SPIKE = json.load(f)
SPIKE_HISTORY = SPIKE["history"]
SPIKE_REPORT = SPIKE["classification_report"]
SPIKE_CM = np.array(SPIKE["confusion_matrix"])

# --- Full MASNN: attention + spiking recurrent layer + temporal attention (Week 4) ---
with open("data/splits/week4_results.json") as f:
    MASNN = json.load(f)
MASNN_HISTORY = MASNN["history"]
MASNN_REPORT = MASNN["classification_report"]
MASNN_CM = np.array(MASNN["confusion_matrix"])

FIG_DPI = 130


def plot_curves_comparison():
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))

    axes[0, 0].plot(OLD_HISTORY["train_loss"], label="train", marker="o")
    axes[0, 0].plot(OLD_HISTORY["val_loss"], label="val", marker="o")
    axes[0, 0].set_title("Broken baseline — Loss\n(lr=1e-3, no class weighting)")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].legend(); axes[0, 0].set_ylim(0, 1.6)

    axes[0, 1].plot(NEW_HISTORY["train_loss"], label="train", marker="o", color="tab:green")
    axes[0, 1].plot(NEW_HISTORY["val_loss"], label="val", marker="o", color="tab:orange")
    axes[0, 1].set_title("Fixed baseline CNN — Loss\n(lr=1e-4 + scheduler, class-weighted, augmented)")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].legend(); axes[0, 1].set_ylim(0, 1.6)

    axes[0, 2].plot(SPIKE_HISTORY["train_loss"], label="train", marker="o", color="tab:purple")
    axes[0, 2].plot(SPIKE_HISTORY["val_loss"], label="val", marker="o", color="tab:brown")
    axes[0, 2].set_title("Spiking network — Loss\n(same fixes, IMG_SIZE=64, 25 time steps)")
    axes[0, 2].set_xlabel("epoch"); axes[0, 2].legend(); axes[0, 2].set_ylim(0, 1.6)

    axes[0, 3].plot(MASNN_HISTORY["train_loss"], label="train", marker="o", color="tab:red")
    axes[0, 3].plot(MASNN_HISTORY["val_loss"], label="val", marker="o", color="tab:cyan")
    axes[0, 3].set_title("MASNN — Loss\n(+ attention + recurrent + temporal attn.)")
    axes[0, 3].set_xlabel("epoch"); axes[0, 3].legend(); axes[0, 3].set_ylim(0, 1.6)

    axes[1, 0].plot(OLD_HISTORY["train_acc"], label="train", marker="o")
    axes[1, 0].plot(OLD_HISTORY["val_acc"], label="val", marker="o")
    axes[1, 0].set_title("Broken baseline — Accuracy")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].legend(); axes[1, 0].set_ylim(0.2, 1.0)

    axes[1, 1].plot(NEW_HISTORY["train_acc"], label="train", marker="o", color="tab:green")
    axes[1, 1].plot(NEW_HISTORY["val_acc"], label="val", marker="o", color="tab:orange")
    axes[1, 1].set_title("Fixed baseline CNN — Accuracy")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].legend(); axes[1, 1].set_ylim(0.2, 1.0)

    axes[1, 2].plot(SPIKE_HISTORY["train_acc"], label="train", marker="o", color="tab:purple")
    axes[1, 2].plot(SPIKE_HISTORY["val_acc"], label="val", marker="o", color="tab:brown")
    axes[1, 2].set_title("Spiking network — Accuracy\n(note: stuck at a constant class for epochs 1-3)")
    axes[1, 2].set_xlabel("epoch"); axes[1, 2].legend(); axes[1, 2].set_ylim(0.2, 1.0)

    axes[1, 3].plot(MASNN_HISTORY["train_acc"], label="train", marker="o", color="tab:red")
    axes[1, 3].plot(MASNN_HISTORY["val_acc"], label="val", marker="o", color="tab:cyan")
    axes[1, 3].set_title("MASNN — Accuracy\n(best epoch 1, val 92%)")
    axes[1, 3].set_xlabel("epoch"); axes[1, 3].legend(); axes[1, 3].set_ylim(0.2, 1.0)

    plt.tight_layout()
    plt.savefig("data/splits/comparison_training_curves.png", dpi=FIG_DPI)
    plt.close(fig)
    print("Saved data/splits/comparison_training_curves.png")


def plot_confusion_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    labels = ["benign", "malignant"]

    for ax, cm, title in [
        (axes[0], OLD_CM, "Broken baseline\n(Week 2, before fix)"),
        (axes[1], NEW_CM, "Fixed baseline CNN\n(Week 2, after fix)"),
        (axes[2], SPIKE_CM, "Spiking network\n(Week 3)"),
        (axes[3], MASNN_CM, "MASNN\n(Week 4)"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    plt.tight_layout()
    plt.savefig("data/splits/comparison_confusion_matrix.png", dpi=FIG_DPI)
    plt.close(fig)
    print("Saved data/splits/comparison_confusion_matrix.png")


def plot_metric_bars():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    metrics = ["benign\nprecision", "benign\nrecall", "malignant\nprecision", "malignant\nrecall", "overall\naccuracy"]
    old_vals = [OLD_REPORT["benign"]["precision"], OLD_REPORT["benign"]["recall"],
                OLD_REPORT["malignant"]["precision"], OLD_REPORT["malignant"]["recall"], OLD_REPORT["accuracy"]]
    new_vals = [NEW_REPORT["benign"]["precision"], NEW_REPORT["benign"]["recall"],
                NEW_REPORT["malignant"]["precision"], NEW_REPORT["malignant"]["recall"], NEW_REPORT["accuracy"]]
    spike_vals = [SPIKE_REPORT["benign"]["precision"], SPIKE_REPORT["benign"]["recall"],
                  SPIKE_REPORT["malignant"]["precision"], SPIKE_REPORT["malignant"]["recall"], SPIKE_REPORT["accuracy"]]
    masnn_vals = [MASNN_REPORT["benign"]["precision"], MASNN_REPORT["benign"]["recall"],
                  MASNN_REPORT["malignant"]["precision"], MASNN_REPORT["malignant"]["recall"], MASNN_REPORT["accuracy"]]

    x = np.arange(len(metrics))
    w = 0.21
    ax.bar(x - 1.5 * w, old_vals, w, label="Broken baseline (Wk2)", color="tab:red", alpha=0.75)
    ax.bar(x - 0.5 * w, new_vals, w, label="Fixed baseline CNN (Wk2)", color="tab:blue", alpha=0.85)
    ax.bar(x + 0.5 * w, spike_vals, w, label="Spiking network (Wk3)", color="tab:green", alpha=0.85)
    ax.bar(x + 1.5 * w, masnn_vals, w, label="MASNN (Wk4)", color="tab:orange", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Test-set performance across all four attempts so far")
    ax.legend()
    for i in range(len(metrics)):
        ax.text(i - 1.5 * w, old_vals[i] + 0.02, f"{old_vals[i]:.2f}", ha="center", fontsize=7)
        ax.text(i - 0.5 * w, new_vals[i] + 0.02, f"{new_vals[i]:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.5 * w, spike_vals[i] + 0.02, f"{spike_vals[i]:.2f}", ha="center", fontsize=7)
        ax.text(i + 1.5 * w, masnn_vals[i] + 0.02, f"{masnn_vals[i]:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig("data/splits/comparison_metric_bars.png", dpi=FIG_DPI)
    plt.close(fig)
    print("Saved data/splits/comparison_metric_bars.png")


if __name__ == "__main__":
    plot_curves_comparison()
    plot_confusion_comparison()
    plot_metric_bars()
    print("\nDone. All comparison images saved to data/splits/")
