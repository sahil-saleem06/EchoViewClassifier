#!/usr/bin/env python3
"""Train the EchoViewClassifier on a labelled image dataset.

Expected data layout:
    data/
      train/
        A2C/  PLAX/  PSAX/  A4C/
      val/
        A2C/  PLAX/  PSAX/  A4C/
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from classifier import EchoViewClassifier, VIEWS, get_dataloader


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_acc += accuracy(logits, labels) * len(labels)
            n += len(labels)
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser(description="Train EchoViewClassifier")
    parser.add_argument("--data", default="data", help="Root directory with train/ and val/ splits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  Views: {VIEWS}")

    train_loader = get_dataloader(args.data, "train", args.batch_size, args.workers)
    val_loader = get_dataloader(args.data, "val", args.batch_size, args.workers)

    num_classes = len(train_loader.dataset.classes)
    model = EchoViewClassifier(num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "classes": train_loader.dataset.classes,
                },
                ckpt_path,
            )
            print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
