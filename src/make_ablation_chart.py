"""
Week 6 — the component-contribution ("ablation-style") chart: each week added
exactly one architectural piece on top of the last, measured on the identical
13-patient test set throughout. This is the project's actual novel research
contribution, per the Research Plan.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

STAGES = [
    ("TSGF +\nBaseline CNN", "week2_results.json"),
    ("+ Spiking\nNeurons", "week3_results.json"),
    ("+ Attention +\nRecurrent Layer", "week4_results.json"),
    ("+ Shrike\nOptimization", "week5_results.json"),
]

labels, acc, benign_recall, malignant_recall = [], [], [], []
for label, fname in STAGES:
    d = json.load(open(f"data/splits/{fname}"))
    r = d["classification_report"]
    labels.append(label)
    acc.append(r["accuracy"])
    benign_recall.append(r["benign"]["recall"])
    malignant_recall.append(r["malignant"]["recall"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(labels))
ax = axes[0]
ax.plot(x, acc, marker="o", markersize=10, linewidth=2.5, color="tab:blue")
for i, v in enumerate(acc):
    ax.annotate(f"{v:.1%}", (i, v), textcoords="offset points", xytext=(0, 12), ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.4, 0.95)
ax.set_ylabel("Test-set accuracy")
ax.set_title("Overall accuracy — one component added per stage")
ax.grid(axis="y", alpha=0.3)

ax2 = axes[1]
ax2.plot(x, benign_recall, marker="o", markersize=9, linewidth=2.5, label="Benign recall", color="tab:orange")
ax2.plot(x, malignant_recall, marker="o", markersize=9, linewidth=2.5, label="Malignant recall", color="tab:green")
for i, v in enumerate(benign_recall):
    ax2.annotate(f"{v:.0%}", (i, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color="tab:orange")
for i, v in enumerate(malignant_recall):
    ax2.annotate(f"{v:.0%}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="tab:green")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylim(0.2, 1.0)
ax2.set_ylabel("Recall")
ax2.set_title("Per-class recall — the precision/recall tradeoff across stages")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Component-contribution analysis: identical test set (1,282 images, 13 patients) at every stage", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("data/splits/ablation_chart.png", dpi=130, bbox_inches="tight")
print("Saved data/splits/ablation_chart.png")

print()
print("Stage-by-stage deltas:")
for i in range(1, len(labels)):
    print(f"  {labels[i-1].replace(chr(10),' ')} -> {labels[i].replace(chr(10),' ')}: "
          f"accuracy {acc[i-1]:.1%} -> {acc[i]:.1%}  ({(acc[i]-acc[i-1])*100:+.1f} points)")
