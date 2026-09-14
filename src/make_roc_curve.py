"""
Week 6 — ROC curve and AUC for the final SOA-refined MASNN, using the saved
week5_model.pt. The paper's own evaluation includes ROC as a metric; we
hadn't computed it for any of our models until now.

The model's raw output is a spike COUNT per class (not a probability), so we
turn it into a proper score via softmax before computing ROC — a standard,
defensible way to get a ranking/confidence signal out of count-based scores.
"""
import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(__file__))
from masnn_model import MASNN
from breakhis_dataset import BreakHisDataset

IMG_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MASNN().to(device)
model.load_state_dict(torch.load("data/splits/week5_model.pt", map_location=device, weights_only=True))
model.eval()

transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
test_ds = BreakHisDataset("data/splits/split.csv", "test", transform=transform, project_root=".")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

all_scores, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)[:, 1]  # probability of "malignant" (class 1)
        all_scores.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC = {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="tab:blue", linewidth=2.5, label=f"MASNN + SOA (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Final Model (Test Set)")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/splits/roc_curve.png", dpi=130)
print("Saved data/splits/roc_curve.png")
