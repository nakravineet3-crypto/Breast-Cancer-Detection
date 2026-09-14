"""
Week 5 — apply the Shrike Optimization Algorithm to refine MASNN's final
classifier layer (fc2: 64 weights + 2 biases = 66 numbers), starting from the
already backprop-trained Week 4 weights. Everything else in the network stays
frozen. Fitness = validation accuracy, evaluated on a small subset (128 images)
for search speed; the final chosen solution is then evaluated properly on the
full validation and test sets.

Saves results to data/splits/week5_results.json.
"""
import sys
import os
import json
import traceback

sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from masnn_model import MASNN
from breakhis_dataset import BreakHisDataset
from shrike_optimizer import ShrikeOptimizer

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG_SIZE = 64
FITNESS_SUBSET_SIZE = 128  # small subset for fast fitness evaluation during search


def flat_to_fc2(model: MASNN, flat: np.ndarray):
    """Load a flat 66-number vector into fc2's weight (64) + bias (2)."""
    w = torch.tensor(flat[:64], dtype=torch.float32).view(2, 32)
    b = torch.tensor(flat[64:66], dtype=torch.float32).view(2)
    with torch.no_grad():
        model.fc2.weight.copy_(w)
        model.fc2.bias.copy_(b)


def fc2_to_flat(model: MASNN) -> np.ndarray:
    w = model.fc2.weight.detach().cpu().numpy().flatten()
    b = model.fc2.bias.detach().cpu().numpy().flatten()
    return np.concatenate([w, b])


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return correct / total, all_preds, all_labels


def negative_loss(model, loader, device):
    """
    Used as the SOA fitness signal instead of accuracy. Accuracy saturates at a
    ceiling (multiple different weight vectors can all score 100% on a small
    subset, giving the search no way to tell them apart — confirmed this
    happened on the first run, fitness pinned at 1.0 from iteration 1 onward).
    Cross-entropy loss is continuous and keeps discriminating between
    candidates even once they all get every prediction right. Negated so
    "higher is better", matching the optimizer's maximize convention.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
    return -(total_loss / total)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    eval_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    split_csv = os.path.join(PROJECT_ROOT, "data/splits/split.csv")

    val_ds = BreakHisDataset(split_csv, "val", transform=eval_transform, project_root=PROJECT_ROOT)
    test_ds = BreakHisDataset(split_csv, "test", transform=eval_transform, project_root=PROJECT_ROOT)

    # Small, fixed subset of validation data used ONLY for fast fitness scoring during search.
    # MUST be randomized, not the first N rows — split.csv is grouped by patient/scan order,
    # and the first 128 rows happen to be 100% benign. A fixed-order subset like that lets the
    # search "cheat" by finding a degenerate always-predict-one-class solution that scores
    # perfectly on the (unrepresentative) subset while failing on the real, balanced data.
    # Confirmed this actually happened on the first run: test accuracy collapsed to 19.5%
    # (benign recall 100%) — almost exactly the benign fraction of the test set.
    rng = np.random.default_rng(42)
    fitness_indices = rng.choice(len(val_ds), size=FITNESS_SUBSET_SIZE, replace=False)
    fitness_loader = DataLoader(torch.utils.data.Subset(val_ds, fitness_indices.tolist()),
                                 batch_size=32, shuffle=False)
    fitness_labels = val_ds.df.iloc[fitness_indices]["label"].value_counts()
    print(f"Fitness subset class balance: {fitness_labels.to_dict()}", flush=True)
    full_val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = MASNN().to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "data/splits/week4_model.pt"), map_location=device))
    print("Loaded Week 4 trained MASNN weights.", flush=True)

    # freeze everything except fc2
    for name, param in model.named_parameters():
        param.requires_grad = ("fc2" in name)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Only these parameters are being searched by SOA:", trainable, flush=True)

    starting_flat = fc2_to_flat(model)
    print(f"fc2 search space: {len(starting_flat)} numbers", flush=True)

    starting_acc, _, _ = evaluate(model, full_val_loader, device)
    starting_test_acc, _, _ = evaluate(model, test_loader, device)
    print(f"Before SOA: val_acc={starting_acc:.4f}  test_acc={starting_test_acc:.4f}", flush=True)

    def fitness_fn(flat_vector: np.ndarray) -> float:
        flat_to_fc2(model, flat_vector)
        return negative_loss(model, fitness_loader, device)

    soa = ShrikeOptimizer(dim=66, n_nests=10, max_iter=15, bounds=(-2.0, 2.0), seed=42)
    best_flat, best_fitness, history = soa.optimize(fitness_fn, init_vector=starting_flat, verbose=True)

    flat_to_fc2(model, best_flat)
    final_val_acc, _, _ = evaluate(model, full_val_loader, device)
    final_test_acc, test_preds, test_labels = evaluate(model, test_loader, device)

    print(f"After SOA:  val_acc={final_val_acc:.4f}  test_acc={final_test_acc:.4f}", flush=True)

    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "data/splits/week5_model.pt"))
    print("Saved final SOA-refined model to week5_model.pt", flush=True)

    report = classification_report(test_labels, test_preds, target_names=["benign", "malignant"], output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()
    print(classification_report(test_labels, test_preds, target_names=["benign", "malignant"]), flush=True)

    results = {
        "fitness_history": history,
        "starting_val_acc": starting_acc,
        "starting_test_acc": starting_test_acc,
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "search_space_size": 66,
        "n_nests": 10,
        "max_iter": 15,
    }
    with open(os.path.join(PROJECT_ROOT, "data/splits/week5_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
