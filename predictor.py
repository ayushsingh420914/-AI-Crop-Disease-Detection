"""Shared loading and prediction functions used by the web app and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from model import build_model

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILENAME = "crop_disease_model.pth"
CLASSES_FILENAME = "classes.json"
CONFIG_FILENAME = "model_config.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_trained_model(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | None = None):
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    classes_path = model_dir / CLASSES_FILENAME
    config_path = model_dir / CONFIG_FILENAME

    missing = [str(path.name) for path in (model_path, classes_path, config_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained model files are missing: " + ", ".join(missing) + ". Run train.py first."
        )

    with classes_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)

    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("classes.json is invalid. Retrain the model.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), use_pretrained=False)
    try:
        state_dict = torch.load(model_path, map_location=resolved_device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state_dict = torch.load(model_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    image_size = int(config.get("image_size", 224))
    image_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return model, class_names, image_transform, resolved_device


def predict_pil_image(
    image: Image.Image,
    model,
    class_names: list[str],
    image_transform,
    device: str,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    if image is None:
        raise ValueError("No image was supplied.")

    image_tensor = image.convert("RGB")
    input_tensor = image_transform(image_tensor).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        count = min(max(1, top_k), len(class_names))
        values, indices = torch.topk(probabilities, k=count)

    results: list[dict[str, float | str]] = []
    for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        results.append({"label": class_names[index], "confidence": float(score)})
    return results
