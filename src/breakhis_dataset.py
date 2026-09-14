"""
PyTorch Dataset for BreakHis, built on top of data/splits/split.csv.
Reused by every week's training notebook from here on.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

LABEL_TO_IDX = {"benign": 0, "malignant": 1}


class BreakHisDataset(Dataset):
    def __init__(self, split_csv: str, split: str, transform=None, project_root: str = ".."):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.project_root = project_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.project_root}/{row['path']}"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = LABEL_TO_IDX[row["label"]]
        return img, label
