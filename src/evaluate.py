"""Evaluate the trained classifier on the held-out test split.

Usage:
    python -m src.evaluate --csv data/splits.csv --ckpt outputs/best_model.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader  # noqa: E402

from src.data import CLASS_NAMES, HistoDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/splits.csv")
    p.add_argument("--ckpt", default="outputs/best_model.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    ds = HistoDataset(args.csv, "test", ckpt["img_size"], ckpt["use_clahe"])
    dl = DataLoader(ds, args.batch_size, shuffle=False)

    probs, labels = [], []
    for x, y in dl:
        probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall_sensitivity": recall_score(labels, preds),
        "specificity": recall_score(labels, preds, pos_label=0),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "n_test": int(len(labels)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)")
    plt.legend()
    plt.savefig(out / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics and plots to {out}/")


if __name__ == "__main__":
    main()
