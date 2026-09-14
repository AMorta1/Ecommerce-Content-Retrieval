from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.data.images import fetch_image, smoke_examples, validate_image


class ImageTests(unittest.TestCase):
    def png(self, size=(96, 128)):
        output = BytesIO()
        Image.new("RGB", size, (20, 60, 100)).save(output, format="PNG")
        return output.getvalue()

    def test_actual_format_and_dimensions(self):
        result = validate_image(self.png())
        self.assertEqual((result["format"], result["width"], result["height"]), ("PNG", 96, 128))
        self.assertEqual(len(result["sha256"]), 64)

    def test_reject_html_small_and_truncated(self):
        for data in (b"<html>not an image</html>", self.png((16, 16)), self.png()[:40]):
            with self.subTest(data=data[:12]), self.assertRaises(Exception):
                validate_image(data)

    def test_current_cache_is_used_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_directory = root / "images"
            image_directory.mkdir()
            (image_directory / "123.png").write_bytes(self.png())
            record = {"product_id": "123", "image_url": "https://img.alicdn.com/123.jpg"}
            with patch("src.data.images.urllib.request.build_opener") as network:
                result = fetch_image(record, root, image_directory, ["img.alicdn.com"])
            self.assertEqual(result["status"], "cached")
            self.assertEqual(result["image_path"], "images/123.png")
            network.assert_not_called()

    def test_examples_use_only_training_and_both_categories(self):
        rows = [{"product_id": str(i), "category_l1": str(i % 2), "category_l2": str(i),
                 "attributes": {"a": ["b"]}, "split": "train" if i < 6 else "test"} for i in range(8)]
        selected = smoke_examples(rows)
        self.assertEqual(len(selected), 5)
        self.assertEqual({r["category_l1"] for r in selected}, {"0", "1"})
        self.assertTrue(all(r["split"] == "train" for r in selected))


if __name__ == "__main__":
    unittest.main()
