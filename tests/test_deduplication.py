from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.data.deduplication import (
    collect_training_exclusions,
    extract_brand_model,
    image_difference_hash,
    normalize_identifier,
    repeated_brand_model_groups,
)


def product(product_id, split, brand, model):
    return {
        "product_id": product_id,
        "split": split,
        "category_l2": "键盘",
        "attributes": {"品牌": [brand], "键盘型号": [model]},
    }


class DeduplicationTests(unittest.TestCase):
    def test_identifier_normalization(self):
        self.assertEqual(normalize_identifier(" HP／惠普 K-500 "), "hp惠普k500")

    def test_placeholder_model_is_ignored(self):
        record = product("1", "train", "Other/其他", "其他/Other")
        self.assertIsNone(extract_brand_model(record))

    def test_unknown_brand_with_specific_model_is_retained(self):
        record = product("1", "train", "Other/其他", "1000000M")
        self.assertEqual(extract_brand_model(record), ("other其他", "1000000m"))

    def test_training_copy_is_excluded_without_touching_frozen_record(self):
        records = [
            product("train-1", "train", "HP/惠普", "K500"),
            product("validation-1", "validation", "hp 惠普", "k-500"),
        ]
        groups = repeated_brand_model_groups(records)
        evidence = collect_training_exclusions(groups, [], {"validation", "test"})
        self.assertEqual(set(evidence), {"train-1"})
        self.assertEqual(evidence["train-1"][0]["reference_id"], "validation-1")

    def test_training_only_duplicates_are_not_cross_split_leakage(self):
        records = [
            product("train-1", "train", "品牌", "A1"),
            product("train-2", "train", "品牌", "A1"),
        ]
        groups = repeated_brand_model_groups(records)
        self.assertEqual(collect_training_exclusions(groups, [], {"validation", "test"}), {})

    def test_difference_hash_is_stable_for_same_image(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.png"
            right = Path(directory) / "right.png"
            image = Image.new("RGB", (20, 20), "white")
            image.putpixel((5, 5), (0, 0, 0))
            image.save(left)
            image.save(right)
            self.assertEqual(image_difference_hash(left), image_difference_hash(right))


if __name__ == "__main__":
    unittest.main()
