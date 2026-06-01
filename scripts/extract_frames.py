#!/usr/bin/env python3
"""Extract frames from EchoNet-Dynamic AVI files for smoke-testing the pipeline.

Samples a fixed number of videos, pulls a few evenly-spaced frames from each,
and writes them into data/train/<class>/ and data/val/<class>/.

Usage:
    python scripts/extract_frames.py \
        --videos /path/to/Videos \
        --output data \
        --class-name A4C \
        --n-train 100 \
        --n-val 20 \
        --frames-per-video 5
"""
import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, n_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return 0

    indices = sorted(random.sample(range(total), n_frames))
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"{video_path.stem}_f{idx:04d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, help="Directory of AVI files")
    parser.add_argument("--output", default="data", help="Root output dir (train/ and val/ created here)")
    parser.add_argument("--class-name", default="A4C")
    parser.add_argument("--n-train", type=int, default=100, help="Number of videos for train split")
    parser.add_argument("--n-val", type=int, default=20, help="Number of videos for val split")
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_videos = sorted(Path(args.videos).glob("*.avi"))
    if len(all_videos) < args.n_train + args.n_val:
        raise ValueError(f"Not enough videos: found {len(all_videos)}, need {args.n_train + args.n_val}")

    selected = random.sample(all_videos, args.n_train + args.n_val)
    train_videos = selected[: args.n_train]
    val_videos = selected[args.n_train :]

    for split, videos in [("train", train_videos), ("val", val_videos)]:
        out_dir = Path(args.output) / split / args.class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        total_saved = 0
        for v in tqdm(videos, desc=f"{split}"):
            total_saved += extract_frames(v, out_dir, args.frames_per_video)
        print(f"{split}: saved {total_saved} frames → {out_dir}")


if __name__ == "__main__":
    main()
