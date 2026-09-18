"""Detect cross-split near duplicates and build a leakage-reduced training set."""

from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata

from PIL import Image


PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "other",
    "其他",
    "其它",
    "other其他",
    "其他other",
    "other其它",
    "其它other",
    "无品牌",
    "无",
    "见详情",
    "不详",
}


def normalize_identifier(value: str) -> str:
    """Normalize brand/model text while retaining Chinese letters and digits."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def first_attribute_value(attributes: dict[str, list[str]], key: str) -> str:
    """Return the first normalized value of a source attribute."""
    values = attributes.get(key, [])
    if not isinstance(values, list) or not values:
        return ""
    return normalize_identifier(values[0])


def extract_brand_model(record: dict) -> tuple[str, str] | None:
    """Extract a usable pair; unknown brands are allowed but invalid models are not."""
    attributes = record.get("attributes", {})
    brand = first_attribute_value(attributes, "品牌")
    model_keys = [key for key in attributes if key == "型号" or key.endswith("型号")]
    model = first_attribute_value(attributes, model_keys[0]) if model_keys else ""
    if not brand or not model or model in PLACEHOLDER_VALUES:
        return None
    return brand, model


def repeated_brand_model_groups(records: list[dict]) -> list[dict]:
    """Return repeated brand/model groups with complete product evidence."""
    grouped = defaultdict(list)
    for record in records:
        brand_model = extract_brand_model(record)
        if brand_model:
            key = (record["category_l2"], *brand_model)
            grouped[key].append(record)

    groups = []
    for (category, brand, model), products in sorted(grouped.items()):
        if len(products) < 2:
            continue
        groups.append(
            {
                "category_l2": category,
                "brand": brand,
                "model": model,
                "products": products,
            }
        )
    return groups


def image_difference_hash(path: Path) -> int:
    """Calculate a 64-bit dHash from a local product image."""
    with Image.open(path) as image:
        pixels = list(
            image.convert("L")
            .resize((9, 8), Image.Resampling.LANCZOS)
            .getdata()
        )
    value = 0
    for y in range(8):
        for x in range(8):
            if pixels[y * 9 + x] > pixels[y * 9 + x + 1]:
                value |= 1 << (y * 8 + x)
    return value


def similar_image_pairs(
    records: list[dict],
    project_root: Path,
    max_distance: int,
) -> list[dict]:
    """Find visually similar pairs inside each fine category."""
    by_category = defaultdict(list)
    for record in records:
        image_path = (project_root / record["image_path"]).resolve()
        image_path.relative_to(project_root.resolve())
        by_category[record["category_l2"]].append(
            (record, image_difference_hash(image_path))
        )

    pairs = []
    for category, products in sorted(by_category.items()):
        for index, (left, left_hash) in enumerate(products):
            for right, right_hash in products[index + 1:]:
                distance = (left_hash ^ right_hash).bit_count()
                if distance <= max_distance:
                    pairs.append(
                        {
                            "category_l2": category,
                            "left": left,
                            "right": right,
                            "distance": distance,
                        }
                    )
    return pairs


def collect_training_exclusions(
    brand_model_groups: list[dict],
    image_pairs: list[dict],
    frozen_splits: set[str],
) -> dict[str, list[dict]]:
    """Map each leaking training product to its frozen-set matches and reasons."""
    evidence = defaultdict(list)
    for group in brand_model_groups:
        frozen = [p for p in group["products"] if p["split"] in frozen_splits]
        training = [p for p in group["products"] if p["split"] == "train"]
        for product in training:
            for reference in frozen:
                evidence[product["product_id"]].append(
                    {
                        "reason": "same_brand_model",
                        "reference_id": reference["product_id"],
                        "reference_split": reference["split"],
                        "detail": f"{group['brand']} / {group['model']}",
                    }
                )

    for pair in image_pairs:
        for product, reference in ((pair["left"], pair["right"]), (pair["right"], pair["left"])):
            if product["split"] == "train" and reference["split"] in frozen_splits:
                evidence[product["product_id"]].append(
                    {
                        "reason": "similar_image",
                        "reference_id": reference["product_id"],
                        "reference_split": reference["split"],
                        "detail": f"dHash distance={pair['distance']}",
                    }
                )
    return dict(evidence)


def category_counts(records: list[dict]) -> dict[str, int]:
    """Return stable fine-category counts for reports."""
    return dict(sorted(Counter(record["category_l2"] for record in records).items()))
