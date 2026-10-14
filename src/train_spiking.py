"""
Standalone training script for the Week 3 spiking classifier (SpikingCNN).
Mirrors train_baseline.py's structure so results are directly comparable:
same class-weighted loss, same LR scheduler, same accuracy-based checkpointing
with early stopping (the criterion we switched to after the Week 2 v1/v2
comparison). Saves results to data/splits/week3_results.json and weights to
data/splits/week3_model.pt.
"""
import sys
import os
import json
import copy
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix

from spiking_model import SpikingCNN, create_healthy_model
from breakhis_dataset import BreakHisDataset

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG_SIZE = 128  # Upgraded to 128x128 for Google Colab GPU training (captures finer cell/nuclear detail)
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 6


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # spike counts per class, shape (batch, 2)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
    ])
    eval_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    split_csv = os.path.join(PROJECT_ROOT, "data/splits/split.csv")

    train_ds = BreakHisDataset(split_csv, "train", transform=train_transform, project_root=PROJECT_ROOT)
    val_ds = BreakHisDataset(split_csv, "val", transform=eval_transform, project_root=PROJECT_ROOT)
    test_ds = BreakHisDataset(split_csv, "test", transform=eval_transform, project_root=PROJECT_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    probe_images, _ = next(iter(train_loader))
    model = create_healthy_model(probe_images).to(device)
    print("Model created OK, params:", sum(p.numel() for p in model.parameters()), flush=True)

    class_counts = train_ds.df["label"].value_counts()
    n_benign, n_malignant = class_counts["benign"], class_counts["malignant"]
    total = n_benign + n_malignant
    weight_benign = total / (2 * n_benign)
    weight_malignant = total / (2 * n_malignant)
    class_weights = torch.tensor([weight_benign, weight_malignant], dtype=torch.float32).to(device)
    print(f"Class weights -> benign={weight_benign:.3f} malignant={weight_malignant:.3f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_val_loss_at_best = None
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_loss_at_best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        flag = "  <- best" if improved else ""
        print(f"Epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} lr={lr_now:.6f}{flag}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)", flush=True)
            break

    model.load_state_dict(best_state)
    print(f"Restored best checkpoint, val_acc={best_val_acc:.4f} (val_loss at that epoch={best_val_loss_at_best:.4f})", flush=True)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    report = classification_report(all_labels, all_preds, target_names=["benign", "malignant"], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    print(classification_report(all_labels, all_preds, target_names=["benign", "malignant"]), flush=True)

    os.makedirs(os.path.join(PROJECT_ROOT, "data/splits"), exist_ok=True)
    results = {
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm,
        "best_val_acc": best_val_acc,
        "best_val_loss_at_best": best_val_loss_at_best,
        "img_size": IMG_SIZE,
        "num_steps": 25,
        "threshold": 0.3,
    }
    with open(os.path.join(PROJECT_ROOT, "data/splits/week3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(best_state, os.path.join(PROJECT_ROOT, "data/splits/week3_model.pt"))
    print("Saved results and model checkpoint.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
