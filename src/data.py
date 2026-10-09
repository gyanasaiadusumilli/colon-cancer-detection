"""Data loading and OpenCV preprocessing for colon histopathology images (LC25000)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Label 0 = benign tissue, label 1 = adenocarcinoma
CLASSES = ["colon_n", "colon_aca"]
CLASS_NAMES = ["Benign", "Adenocarcinoma"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel (LAB space).

    H&E stained slides vary a lot in staining intensity between labs and scanners.
    Equalizing only the lightness channel boosts local contrast of nuclei and gland
    structures without distorting the stain colors.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(img_bgr: np.ndarray, size: int = 224, use_clahe: bool = True) -> np.ndarray:
    """BGR uint8 image -> RGB float32 in [0, 1], resized to (size, size)."""
    if use_clahe:
        img_bgr = apply_clahe(img_bgr)
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentations that make sense for tissue patches (no fixed orientation)."""
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.5:
        img = np.flipud(img)
    img = np.rot90(img, k=int(rng.integers(0, 4)))
    # Mild brightness / contrast jitter to simulate staining variation
    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-0.05, 0.05)
    img = np.clip(img * alpha + beta, 0.0, 1.0)
    return np.ascontiguousarray(img)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC float [0,1] -> normalized CHW tensor."""
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


class HistoDataset(Dataset):
    def __init__(self, csv_path: str, split: str, size: int = 224,
                 use_clahe: bool = True, train: bool = False, seed: int = 42):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split '{split}' in {csv_path}")
        self.size = size
        self.use_clahe = use_clahe
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(Path(row["path"])))
        if img_bgr is None:
            raise FileNotFoundError(row["path"])
        img = preprocess(img_bgr, self.size, self.use_clahe)
        if self.train:
            img = augment(img, self.rng)
        return to_tensor(img), int(row["label"])
