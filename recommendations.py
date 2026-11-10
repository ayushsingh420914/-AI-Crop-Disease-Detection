"""Simple, cautious integrated-pest-management guidance for predicted labels."""

from __future__ import annotations


def format_label(label: str) -> tuple[str, str]:
    """Convert common PlantVillage folder names into crop and condition text."""
    cleaned = label.replace("___", " - ").replace("_", " ").strip()
    if " - " in cleaned:
        crop, condition = cleaned.split(" - ", 1)
    else:
        crop, condition = "Crop", cleaned
    return crop, condition


def get_recommendations(label: str) -> dict[str, object]:
    """Return general next steps; never prescribe a pesticide or dosage."""
    crop, condition = format_label(label)
    text = condition.lower()

    if "healthy" in text:
        return {
            "crop": crop,
            "condition": condition,
            "status": "No disease pattern detected",
            "actions": [
                "Continue regular field scouting and inspect both sides of leaves.",
                "Use clean tools and healthy planting material.",
                "Maintain suitable irrigation, drainage, spacing, and balanced nutrition.",
            ],
        }

    if any(word in text for word in ("virus", "mosaic", "leaf curl", "yellow leaf")):
        actions = [
            "Inspect nearby plants for similar symptoms and mark affected plants.",
            "Remove severely affected plants only where local agricultural guidance recommends it; do not compost plants suspected of carrying a virus.",
            "Check for insect vectors such as aphids or whiteflies and use locally approved integrated pest-management practices.",
            "Clean tools and avoid moving from affected plants to healthy plants without cleaning.",
        ]
    elif "bacterial" in text:
        actions = [
            "Avoid handling plants while foliage is wet; water near the soil rather than overhead.",
            "Remove badly affected plant debris where practical and dispose of it safely.",
            "Disinfect pruning tools between plants and avoid working in wet fields.",
            "Ask a local KVK/agriculture officer to confirm the diagnosis before choosing any treatment.",
        ]
    elif any(word in text for word in ("blight", "mildew", "mold", "mould", "rust", "scab", "rot", "spot", "septoria", "anthracnose")):
        actions = [
            "Remove heavily affected leaves or fallen crop debris where practical; do not leave infected residues in the field.",
            "Improve plant spacing and airflow, and avoid prolonged leaf wetness.",
            "Water at the base of plants and maintain good field drainage.",
            "Confirm the disease locally before using a fungicide; follow the product label and local agricultural advice exactly.",
        ]
    else:
        actions = [
            "Inspect several plants and compare symptoms on young and mature leaves.",
            "Remove badly affected debris where practical and keep tools clean.",
            "Avoid overwatering and improve airflow around plants.",
            "Take the image and a leaf sample to your nearest KVK/agriculture extension officer for confirmation.",
        ]

    return {
        "crop": crop,
        "condition": condition,
        "status": "Possible disease pattern detected",
        "actions": actions,
    }


DISCLAIMER = (
    "This is an educational AI screening tool, not a laboratory diagnosis. "
    "Leaf-image predictions can be wrong, especially with poor lighting, mixed symptoms, "
    "or field conditions. Confirm the result with a qualified local agriculture/KVK expert "
    "before taking crop-protection or chemical-treatment decisions."
)
