"""Train a colon histopathology classifier.

Usage:
    python -m src.train --csv data/splits.csv --arch resnet18 --epochs 10
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import HistoDataset
from src.model import build_model
from src.utils import get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/splits.csv")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing (for ablation)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_clahe = not args.no_clahe

    train_ds = HistoDataset(args.csv, "train", args.img_size, use_clahe, train=True, seed=args.seed)
    val_ds = HistoDataset(args.csv, "val", args.img_size, use_clahe)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device} | train={len(train_ds)} val={len(val_ds)} | arch={args.arch} clahe={use_clahe}")
    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "arch": args.arch, "img_size": args.img_size,
                        "use_clahe": use_clahe}, out / "best_model.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val accuracy: {best_acc:.4f}. Checkpoint saved to {out / 'best_model.pt'}")


if __name__ == "__main__":
    main()
