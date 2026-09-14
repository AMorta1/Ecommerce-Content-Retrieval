import tempfile
import unittest
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.validate_data import write_category_review
from src.data.review import find_product_id_mismatches, product_id_from_image_path, restore_product_ids


class ReviewSpreadsheetTests(unittest.TestCase):
    def test_existing_review_labels_are_not_overwritten(self) -> None:
        records = []
        for category_index in range(2):
            for item_index in range(5):
                product_id = str(category_index * 10 + item_index)
                records.append(
                    {
                        "product_id": product_id,
                        "category_l1": "测试大类",
                        "category_l2": f"测试品类{category_index}",
                        "raw_label": "测试标签",
                        "title": "测试商品",
                        "attributes": {"型号": [product_id]},
                        "image_path": f"data/images/{product_id}.jpg",
                        "split": "train",
                    }
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            review_path = Path(temporary_directory) / "review.csv"
            with redirect_stdout(StringIO()):
                write_category_review(records, review_path)
            with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[0]["review_label_ok"] = "1"
            with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with redirect_stdout(StringIO()):
                write_category_review(records, review_path)

            with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
                preserved = list(csv.DictReader(handle))
            self.assertEqual(preserved[0]["review_label_ok"], "1")

    def test_product_id_is_read_from_image_filename(self) -> None:
        result = product_id_from_image_path("data/images/week1_v3/610580284930.webp")

        self.assertEqual(result, "610580284930")

    def test_scientific_notation_id_can_be_restored(self) -> None:
        rows = [
            {
                "product_id": "6.1058E+11",
                "image_path": "data/images/week1_v3/610580284930.webp",
            }
        ]

        mismatches = find_product_id_mismatches(rows)
        repaired = restore_product_ids(rows)

        self.assertEqual(mismatches, [(2, "6.1058E+11", "610580284930")])
        self.assertEqual(repaired, 1)
        self.assertEqual(rows[0]["product_id"], "610580284930")

    def test_invalid_image_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "placeholder.webp"

            with self.assertRaisesRegex(ValueError, "不是数字商品编号"):
                product_id_from_image_path(str(path))


if __name__ == "__main__":
    unittest.main()
