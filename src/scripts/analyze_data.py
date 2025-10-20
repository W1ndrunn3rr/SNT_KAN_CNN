"""Utility script for inspecting dataset structure and spotting distribution issues."""
from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class DatasetStats:
    root: Path
    num_samples: int
    classes: Sequence[str]
    class_counts: Counter[str]
    mean: np.ndarray
    std: np.ndarray
    widths: list[int]
    heights: list[int]


RGB_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
RGB_NORMALIZE_STD = [0.229, 0.224, 0.225]
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image dataset statistics")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/train"),
        help="Path to training data root (default: data/train)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/test"),
        help="Path to test data root (default: data/test)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="How many images to sample for statistics (default: 512)",
    )
    parser.add_argument(
        "--max-duplicate-check",
        type=int,
        default=1024,
        help="Max number of images per split for duplicate hash checking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory, got file: {path}")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_NORMALIZE_MEAN, std=RGB_NORMALIZE_STD),
        ]
    )


def collect_stats(root: Path, sample_size: int, seed: int) -> DatasetStats:
    ensure_dir(root)
    transform = build_transform()
    dataset = ImageFolder(root=str(root), transform=transform)

    counts = Counter()
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1

    rng = np.random.default_rng(seed)
    total = len(dataset)
    actual_sample_size = min(sample_size, total)
    indices = rng.choice(total, size=actual_sample_size, replace=False)

    means = []
    stds = []
    widths: list[int] = []
    heights: list[int] = []

    for idx in indices:
        tensor, _ = dataset[idx]
        means.append(tensor.mean(dim=(1, 2)).cpu().numpy())
        stds.append(tensor.std(dim=(1, 2)).cpu().numpy())

        image_path = Path(dataset.samples[idx][0])
        with Image.open(image_path) as img:
            width, height = img.size
        widths.append(width)
        heights.append(height)

    mean = np.stack(means).mean(axis=0) if means else np.zeros(3)
    std = np.stack(stds).mean(axis=0) if stds else np.zeros(3)

    return DatasetStats(
        root=root,
        num_samples=total,
        classes=dataset.classes,
        class_counts=counts,
        mean=mean,
        std=std,
        widths=widths,
        heights=heights,
    )


def iter_image_paths(root: Path) -> Iterable[Path]:
    yield from itertools.chain.from_iterable(
        root.glob(f"**/*{suffix}") for suffix in SUPPORTED_SUFFIXES
    )


def compute_hashes(paths: Iterable[Path], limit: int) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for count, path in enumerate(paths):
        if count >= limit:
            break
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        mapping[digest] = path
    return mapping


def report_stats(stats: DatasetStats, name: str) -> None:
    print("=" * 70)
    print(f"📊 {name} ({stats.root})")
    print("=" * 70)
    print(f"Total samples: {stats.num_samples}")
    print(f"Classes ({len(stats.classes)}): {stats.classes}")
    print("Class distribution:")
    for label in sorted(stats.class_counts):
        count = stats.class_counts[label]
        ratio = (count / max(stats.num_samples, 1)) * 100
        print(f"  - {label:<15} {count:6d} ({ratio:5.2f}%)")

    if stats.widths and stats.heights:
        print("Image size (original pixels):")
        print(
            f"  Width  min={min(stats.widths):4d}  max={max(stats.widths):4d}  "
            f"mean={np.mean(stats.widths):7.2f}"
        )
        print(
            f"  Height min={min(stats.heights):4d}  max={max(stats.heights):4d}  "
            f"mean={np.mean(stats.heights):7.2f}"
        )

    if stats.mean.size == 3 and stats.std.size == 3:
        print("Normalized tensor statistics (per channel):")
        print(
            "  Mean : "
            f"R={stats.mean[0]:.4f}, G={stats.mean[1]:.4f}, B={stats.mean[2]:.4f}"
        )
        print(
            "  Std  : "
            f"R={stats.std[0]:.4f}, G={stats.std[1]:.4f}, B={stats.std[2]:.4f}"
        )
    print()


def detect_duplicates(train_dir: Path, test_dir: Path, limit: int) -> None:
    print("=" * 70)
    print("🔍 Checking for potential duplicates between train and test splits")
    print("=" * 70)
    train_hashes = compute_hashes(iter_image_paths(train_dir), limit)
    test_hashes = compute_hashes(iter_image_paths(test_dir), limit)
    overlap = set(train_hashes).intersection(test_hashes)

    if not overlap:
        print(
            f"No duplicates detected within the first {limit} files per split.\n"
        )
        return

    print(f"⚠️ Found {len(overlap)} duplicates (showing up to 5):")
    for digest in list(overlap)[:5]:
        print("  - Train:", train_hashes[digest])
        print("    Test :", test_hashes[digest])
    print()


def main() -> None:
    args = parse_args()

    train_stats = collect_stats(args.train_dir, args.sample_size, args.seed)
    report_stats(train_stats, "TRAIN")

    if args.test_dir.exists():
        test_stats = collect_stats(args.test_dir, args.sample_size, args.seed)
        report_stats(test_stats, "TEST")

        class_gap = set(train_stats.classes) ^ set(test_stats.classes)
        if class_gap:
            print("⚠️ Class mismatch detected between splits:")
            for name in sorted(class_gap):
                location = "only in TEST" if name in test_stats.classes else "only in TRAIN"
                print(f"  - {name} ({location})")
            print()

        detect_duplicates(args.train_dir, args.test_dir, args.max_duplicate_check)
    else:
        print(f"Test directory {args.test_dir} not found; skipping test split analysis.\n")


if __name__ == "__main__":
    main()
