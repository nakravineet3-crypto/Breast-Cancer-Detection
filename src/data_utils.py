"""
Colab-friendly data loading: scan wherever the dataset lives (e.g. a Kaggle
download), then attach the committed patient-wise split by *filename*, so the
split is identical to earlier experiments regardless of folder layout.
Images are resized once and cached in RAM (fast epochs, few workers needed).
"""
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(__file__))
from build_dataset import scan_dataset  # noqa: E402

LABEL_TO_IDX = {"benign": 0, "malignant": 1}


def build_manifest(data_root: str, split_csv: str) -> pd.DataFrame:
    df = scan_dataset(data_root)
    if df.empty:
        raise RuntimeError(f"No BreakHis PNGs matched under {data_root}")
    df["fname"] = df["path"].map(os.path.basename)
    df = df.drop_duplicates("fname")  # some Kaggle uploads nest a second copy of the folder
    ref = pd.read_csv(split_csv)
    ref["fname"] = ref["path"].map(os.path.basename)
    df = df.merge(ref[["fname", "split"]], on="fname", how="inner")
    print(f"[data] found {len(df)} images with a split (reference split has {len(ref)})")
    if len(df) < len(ref):
        print(f"[data] WARNING: {len(ref) - len(df)} reference images not found under {data_root}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    return df.reset_index(drop=True)


class CachedImages(Dataset):
    """Loads + resizes every image once; applies `transform` (PIL -> tensor) lazily."""
    def __init__(self, df: pd.DataFrame, cache_size: int, transform=None):
        self.labels = [LABEL_TO_IDX[l] for l in df["label"]]
        self.mags = list(df["magnification"])
        self.transform = transform
        self.imgs = []
        for i, p in enumerate(df["path"]):
            self.imgs.append(Image.open(p).convert("RGB").resize((cache_size, cache_size), Image.BILINEAR))
            if (i + 1) % 2000 == 0:
                print(f"[data] cached {i + 1}/{len(df)}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


@torch.no_grad()
def predict_proba(model, loader, device, tta: bool = False):
    """Returns P(malignant) per image; optional flip-TTA (orig, hflip, vflip averaged)."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        views = [x, x.flip(3), x.flip(2)] if tta else [x]
        p = sum(torch.softmax(model(v).float(), dim=1)[:, 1] for v in views) / len(views)
        out.append(p.cpu())
    return torch.cat(out).numpy()
