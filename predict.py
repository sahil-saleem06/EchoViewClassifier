#!/usr/bin/env python3
"""Run inference on a single image or a directory of images.

Usage:
    python predict.py --checkpoint checkpoints/best.pt --input image.png
    python predict.py --checkpoint checkpoints/best.pt --input frames/
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from classifier import EchoViewClassifier
from classifier.transforms import get_val_transforms


def predict_image(model, transform, image_path: str, device: str, classes: list[str]) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    return {
        "file": str(image_path),
        "predicted_view": classes[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description="Echo view inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes")
    if classes is None:
        from classifier import VIEWS
        classes = sorted(VIEWS)

    model = EchoViewClassifier.from_checkpoint(args.checkpoint, device=device)
    transform = get_val_transforms()

    input_path = Path(args.input)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    if input_path.is_dir():
        paths = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in image_exts]
        if not paths:
            print(f"No images found in {input_path}", file=sys.stderr)
            sys.exit(1)
        results = [predict_image(model, transform, p, device, classes) for p in paths]
        print(json.dumps(results, indent=2))
    else:
        result = predict_image(model, transform, input_path, device, classes)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
