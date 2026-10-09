"""Build a stratified train/val/test split CSV from the LC25000 colon folders.

Usage:
    python scripts/split_data.py --src data/lung_colon_image_set/colon_image_sets --out data/splits.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder containing colon_aca/ and colon_n/")
    p.add_argument("--out", default="data/splits.csv")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = []
    for label, cls in enumerate(CLASSES):
        folder = Path(args.src) / cls
        if not folder.is_dir():
            sys.exit(f"Missing folder: {folder}")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(f.resolve()), "label": label})
    df = pd.DataFrame(rows)

    train_val, test = train_test_split(df, test_size=args.test, stratify=df["label"],
                                       random_state=args.seed)
    val_frac = args.val / (1 - args.test)
    train, val = train_test_split(train_val, test_size=val_frac, stratify=train_val["label"],
                                  random_state=args.seed)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")

    out = pd.concat([train, val, test]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["split", "label"]).size().unstack())
    print(f"Saved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
