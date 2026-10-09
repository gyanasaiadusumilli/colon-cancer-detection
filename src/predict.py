"""Predict a single histopathology image.

Usage:
    python -m src.predict path/to/image.jpeg --ckpt outputs/best_model.pt
"""
import argparse

import cv2
import numpy as np
import torch

from src.data import CLASS_NAMES, preprocess, to_tensor
from src.evaluate import load_checkpoint
from src.utils import get_device


@torch.no_grad()
def predict_image(img_bgr: np.ndarray, model, ckpt, device) -> dict:
    img = preprocess(img_bgr, ckpt["img_size"], ckpt["use_clahe"])
    x = to_tensor(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {"label": CLASS_NAMES[idx], "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--ckpt", default="outputs/best_model.pt")
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    result = predict_image(img, model, ckpt, device)
    print(f"{result['label']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
