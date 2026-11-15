"""
Week 6 MASNN v2 training (built for unattended Colab runs).

  python src/train_masnn_v2.py --data-root /path/to/BreaKHis --out-dir /content/drive/MyDrive/masnn_out --seeds 0 1 2

- Same patient-wise split as before (data/splits/split.csv, matched by filename).
- Checkpoint selection on validation ROC-AUC.
- Resumable: per-epoch checkpoint; finished seeds are skipped on rerun.
- Writes <out-dir>/week6_results.json, week6_roc.png, per-seed logs/models.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from masnn_model import MASNNv2
from data_utils import build_manifest, CachedImages, predict_proba
from metrics_utils import safe_auc, summarize, youden_threshold, per_magnification_auc, save_roc_plot

ROOT = os.path.join(os.path.dirname(__file__), "..")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-csv", default=os.path.join(ROOT, "data/splits/split.csv"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=15, help="SNN time steps T")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (off by default; snnTorch membranes can be touchy)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    return ap.parse_args()


def make_loaders(df, args):
    S = args.img_size
    train_tf = T.Compose([
        T.RandomResizedCrop(S, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(0.2, 0.2, 0.2, 0.03),  # stain variation
        T.ToTensor(),
    ])
    eval_tf = T.ToTensor()
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if args.smoke:
        parts = {s: p.groupby("label").head(16 if s == "train" else 8).reset_index(drop=True) for s, p in parts.items()}
    ds = {
        "train": CachedImages(parts["train"], int(S * 1.15), train_tf),
        "val": CachedImages(parts["val"], S, eval_tf),
        "test": CachedImages(parts["test"], S, eval_tf),
    }
    kw = dict(num_workers=args.workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    loaders = {
        "train": DataLoader(ds["train"], args.batch_size, shuffle=True, drop_last=True, **kw),
        "val": DataLoader(ds["val"], args.batch_size, **kw),
        "test": DataLoader(ds["test"], args.batch_size, **kw),
    }
    return ds, loaders


def run_seed(seed, args, ds, loaders, device, log):
    sdir = os.path.join(args.out_dir, f"seed{seed}")
    os.makedirs(sdir, exist_ok=True)
    final_path = os.path.join(sdir, "preds.npz")
    if os.path.exists(final_path):
        log(f"[seed {seed}] already finished, loading preds")
        return dict(np.load(final_path))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MASNNv2(num_steps=args.steps).to(device)
    log(f"[seed {seed}] params: {sum(p.numel() for p in model.parameters()):,}")

    labels = np.array(ds["train"].labels)
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    ckpt_path = os.path.join(sdir, "last.pt")
    best_path = os.path.join(sdir, "best.pt")
    state = {"epoch": 0, "best_auc": -1.0, "best_bal": -1.0, "bad": 0, "history": []}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"[seed {seed}] resumed from epoch {state['epoch']}")

    for epoch in range(state["epoch"], args.epochs):
        t0 = time.time()
        model.train()
        tot, n, correct = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(x)
                loss = criterion(out.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()

        pv = predict_proba(model, loaders["val"], device)
        yv = np.array(ds["val"].labels)
        v_auc = safe_auc(yv, pv)
        v_bal = summarize(yv, pv)["balanced_accuracy"]
        state["history"].append({"epoch": epoch + 1, "train_loss": tot / n, "train_acc": correct / n,
                                 "val_auc": v_auc, "val_bal_acc": v_bal})
        improved = (v_auc, v_bal) > (state["best_auc"], state["best_bal"])
        if improved:
            state.update(best_auc=v_auc, best_bal=v_bal, bad=0)
            torch.save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        state["epoch"] = epoch + 1
        log(f"[seed {seed}] ep {epoch+1}/{args.epochs} loss {tot/n:.4f} train_acc {correct/n:.3f} "
            f"val_auc {v_auc:.4f} val_bal {v_bal:.3f} {'*' if improved else ''} ({time.time()-t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, ckpt_path)
        if state["bad"] >= args.patience:
            log(f"[seed {seed}] early stop"); break

    model.load_state_dict(torch.load(best_path, map_location=device))
    preds = {
        "val": predict_proba(model, loaders["val"], device, tta=True),
        "test": predict_proba(model, loaders["test"], device, tta=True),
        "test_notta": predict_proba(model, loaders["test"], device, tta=False),
    }
    np.savez(final_path, **preds)
    json.dump(state["history"], open(os.path.join(sdir, "history.json"), "w"), indent=1)
    return preds


def main():
    args = parse()
    if args.smoke:
        args.img_size, args.steps, args.epochs, args.batch_size, args.workers = 32, 3, 1, 8, 0
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out_dir, "train_log.txt"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"device={device} args={vars(args)}")
    df = build_manifest(args.data_root, args.split_csv)
    ds, loaders = make_loaders(df, args)
    yv, yt = np.array(ds["val"].labels), np.array(ds["test"].labels)
    mags = ds["test"].mags

    all_preds, per_seed = [], {}
    for seed in args.seeds:
        pr = run_seed(seed, args, ds, loaders, device, log)
        all_preds.append(pr)
        thr = youden_threshold(yv, pr["val"])
        per_seed[seed] = {
            "test_auc_tta": safe_auc(yt, pr["test"]),
            "test_auc_no_tta": safe_auc(yt, pr["test_notta"]),
            "val_auc": safe_auc(yv, pr["val"]),
            "at_0.5": summarize(yt, pr["test"], 0.5),
            "at_val_youden": summarize(yt, pr["test"], thr),
            "per_magnification_auc": per_magnification_auc(yt, pr["test"], mags),
        }
        log(f"[seed {seed}] TEST AUC {per_seed[seed]['test_auc_tta']:.4f} | "
            f"bal-acc@youden {per_seed[seed]['at_val_youden']['balanced_accuracy']:.3f}")

    ens_val = np.mean([p["val"] for p in all_preds], axis=0)
    ens_test = np.mean([p["test"] for p in all_preds], axis=0)
    thr = youden_threshold(yv, ens_val)
    aucs = [v["test_auc_tta"] for v in per_seed.values()]
    results = {
        "config": vars(args),
        "per_seed": per_seed,
        "mean_test_auc": float(np.mean(aucs)), "std_test_auc": float(np.std(aucs)),
        "ensemble": {
            "test_auc": safe_auc(yt, ens_test), "val_auc": safe_auc(yv, ens_val),
            "at_0.5": summarize(yt, ens_test, 0.5),
            "at_val_youden": summarize(yt, ens_test, thr),
            "per_magnification_auc": per_magnification_auc(yt, ens_test, mags),
        },
    }
    json.dump(results, open(os.path.join(args.out_dir, "week6_results.json"), "w"), indent=2)
    save_roc_plot(yt, ens_test, os.path.join(args.out_dir, "week6_roc.png"))
    log("\n===== RESULTS (paste this back) =====")
    log(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
