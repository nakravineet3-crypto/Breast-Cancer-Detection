"""
Scans the raw BreakHis folder tree and produces a single manifest CSV
(one row per image) plus a patient-wise train/val/test split.

Run:  venv\\Scripts\\python.exe src\\build_dataset.py
"""
import os
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_ROOT = "data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
OUT_MANIFEST = "data/splits/manifest.csv"
OUT_SPLIT = "data/splits/split.csv"

FILENAME_RE = re.compile(
    r"SOB_(?P<class_code>[BM])_(?P<subtype>[A-Z]+)-(?P<pid1>\d+)-(?P<pid2>[\w]+)-(?P<mag>\d+)-(?P<seq>\d+)\.png",
    re.IGNORECASE,
)


def scan_dataset(root: str) -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            g = m.groupdict()
            patient_id = f"{g['pid1']}-{g['pid2']}"
            rows.append({
                "path": os.path.join(dirpath, fname).replace("\\", "/"),
                "label": "benign" if g["class_code"].upper() == "B" else "malignant",
                "subtype": g["subtype"],
                "magnification": f"{g['mag']}X",
                "patient_id": patient_id,
            })
    return pd.DataFrame(rows)


def patient_wise_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Splits by patient_id (not by image) so no patient's tissue appears
    in more than one of train/val/test. Matches the paper's 70/15/15 split.
    """
    df = df.copy()
    df["split"] = ""

    # Split off test set first (15% of patients), then val (15% of the remainder's patients)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df["patient_id"]))

    trainval_df = df.iloc[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)  # 0.15/0.85 ≈ 0.1765
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["patient_id"]))

    df.iloc[trainval_idx[train_idx], df.columns.get_loc("split")] = "train"
    df.iloc[trainval_idx[val_idx], df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df


def main():
    print(f"Scanning {RAW_ROOT} ...")
    df = scan_dataset(RAW_ROOT)
    print(f"Found {len(df)} images from {df['patient_id'].nunique()} patients")

    os.makedirs("data/splits", exist_ok=True)
    df.to_csv(OUT_MANIFEST, index=False)

    df = patient_wise_split(df)
    df.to_csv(OUT_SPLIT, index=False)

    print("\nSplit summary (images):")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    print("\nSplit summary (patients, verifying no overlap):")
    for split_name in ["train", "val", "test"]:
        pids = set(df[df.split == split_name]["patient_id"])
        print(f"  {split_name}: {len(pids)} patients")

    # sanity check: no patient appears in more than one split
    overlap = set()
    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            pa = set(df[df.split == a]["patient_id"])
            pb = set(df[df.split == b]["patient_id"])
            overlap |= pa & pb
    assert not overlap, f"Patient leakage across splits! {overlap}"
    print("\nNo patient leakage between splits — confirmed.")
    print(f"\nSaved manifest -> {OUT_MANIFEST}")
    print(f"Saved split     -> {OUT_SPLIT}")


if __name__ == "__main__":
    main()
