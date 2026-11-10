"""Predict the likely crop/leaf condition for one image from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from predictor import DEFAULT_MODEL_DIR, load_trained_model, predict_pil_image
from recommendations import DISCLAIMER, get_recommendations


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf photo.")
    parser.add_argument("image", help="Path to a JPG/PNG/WebP leaf image.")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")

    model, class_names, image_transform, device = load_trained_model(args.model_dir)
    with Image.open(image_path) as image:
        predictions = predict_pil_image(image, model, class_names, image_transform, device)

    best = predictions[0]
    print(f"\nPrediction: {best['label']}")
    print(f"Model confidence score: {best['confidence']:.2%}\n")
    print("Top predictions:")
    for item in predictions:
        print(f"  {item['label']}: {item['confidence']:.2%}")

    advice = get_recommendations(str(best["label"]))
    print(f"\nCrop: {advice['crop']}")
    print(f"Detected pattern: {advice['condition']}")
    print("Suggested next steps:")
    for action in advice["actions"]:
        print(f"  - {action}")
    print(f"\nNote: {DISCLAIMER}")


if __name__ == "__main__":
    main()
