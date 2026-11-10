"""Model factory for AI Yield: Crop Disease Detection and Management."""

from __future__ import annotations

import warnings

from torch import nn
from torchvision import models


def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    """Create a MobileNetV3-Small classifier.

    Pretrained ImageNet weights are used when requested and downloadable.
    If downloading fails, the model safely falls back to random initialization.
    """
    if num_classes < 2:
        raise ValueError("At least two disease/health classes are required.")

    weights = None
    if use_pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except AttributeError:
            warnings.warn(
                "This torchvision version does not expose MobileNetV3 weights. "
                "Continuing with random initialization.",
                RuntimeWarning,
            )
        else:
            # The weights download happens when the model is constructed.
            # Do not stop the project from running when the machine is offline.
            try:
                model = models.mobilenet_v3_small(weights=weights)
            except Exception as exc:  # noqa: BLE001 - network/cache errors vary by platform
                warnings.warn(
                    f"Could not load pretrained weights ({exc}). "
                    "Continuing with random initialization.",
                    RuntimeWarning,
                )
                model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_small(weights=None)

    if weights is None and use_pretrained:
        model = models.mobilenet_v3_small(weights=None)

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model
