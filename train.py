"""Train a MobileNetV3-Small crop disease classifier on an ImageFolder dataset."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model
from predictor import (
    CLASSES_FILENAME,
    CONFIG_FILENAME,
    DEFAULT_MODEL_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_FILENAME,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _direct_image_count(folder: Path) -> int:
    try:
        return sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    except PermissionError:
        return 0


def find_imagefolder_root(data_dir: str | Path) -> Path:
    """Find the folder whose immediate child folders contain class images."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = [root]
    try:
        candidates.extend(path for path in root.rglob("*") if path.is_dir())
    except PermissionError:
        pass

    best_path: Path | None = None
    best_class_count = 0
    for candidate in candidates:
        try:
            class_dirs = [
                child for child in candidate.iterdir()
                if child.is_dir() and _direct_image_count(child) > 0
            ]
        except (PermissionError, OSError):
            continue

        if len(class_dirs) > best_class_count:
            best_path = candidate
            best_class_count = len(class_dirs)

    if best_path is None or best_class_count < 2:
        raise FileNotFoundError(
            "Could not find a dataset folder with at least two class subfolders containing images.\n"
            "Expected: data/raw/PlantVillage/Tomato___Early_blight/image1.jpg"
        )
    print(f"Using dataset folder: {best_path}")
    print(f"Detected class folders: {best_class_count}")
    return best_path


def stratified_split(targets: list[int], validation_fraction: float, seed: int):
    """Create a deterministic per-class train/validation split."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_index, indices in grouped.items():
        if len(indices) < 2:
            raise ValueError(
                f"Class index {class_index} has fewer than 2 images. "
                "Each class needs at least 2 images for a train/validation split."
            )
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=18),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, validation_transform


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("The data loader returned no images.")
    return running_loss / total, correct / total


def train(args) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training device: {device}")

    dataset_root = find_imagefolder_root(args.data_dir)
    train_transform, validation_transform = make_transforms(args.image_size)
    train_dataset_full = datasets.ImageFolder(dataset_root, transform=train_transform)
    validation_dataset_full = datasets.ImageFolder(dataset_root, transform=validation_transform)

    if train_dataset_full.classes != validation_dataset_full.classes:
        raise RuntimeError("Training and validation class mappings do not match.")
    if len(train_dataset_full.classes) < 2:
        raise ValueError("The dataset must contain at least two classes.")

    train_indices, validation_indices = stratified_split(
        train_dataset_full.targets, args.validation_fraction, args.seed
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    validation_dataset = Subset(validation_dataset_full, validation_indices)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    class_counts = np.bincount(
        np.asarray(train_dataset_full.targets)[train_indices],
        minlength=len(train_dataset_full.classes),
    )
    class_weights = len(train_indices) / (len(class_counts) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = build_model(
        num_classes=len(train_dataset_full.classes),
        use_pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    best_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(f"Classes ({len(train_dataset_full.classes)}): {', '.join(train_dataset_full.classes)}")
    print(f"Training images: {len(train_dataset)} | Validation images: {len(validation_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "validation_loss": round(val_loss, 6),
            "validation_accuracy": round(val_accuracy, 6),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.2%} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.2%}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping: validation accuracy did not improve.")
            break

    with (model_dir / CLASSES_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(train_dataset_full.classes, file, indent=2)
    with (model_dir / CONFIG_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "architecture": "mobilenet_v3_small",
                "image_size": args.image_size,
                "validation_fraction": args.validation_fraction,
                "best_validation_accuracy": best_accuracy,
                "dataset_directory": str(dataset_root),
                "seed": args.seed,
            },
            file,
            indent=2,
        )
    with (model_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print("Next step: streamlit run app.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the AI Yield crop disease classifier."
    )
    parser.add_argument("--data_dir", default="data/raw", help="Dataset folder or its parent folder.")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="Where model files are saved.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for safest Windows compatibility.")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet starting weights.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        parser.error("epochs/batch_size must be positive and image_size must be at least 64.")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("validation_fraction must be between 0.05 and 0.5.")
    return args


if __name__ == "__main__":
    warnings.filterwarnings("default")
    train(parse_args())
