import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.catalog import load_catalog, load_metadata, write_metadata
from src.retrieval.ernie_vil import ErnieVilEmbedder, normalize_rows


class RetrievalCatalogTests(unittest.TestCase):
    def test_catalog_and_metadata_preserve_vector_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "image.webp"
            image_path.write_bytes(b"placeholder")
            dataset_path = root / "products.jsonl"
            records = [
                {
                    "product_id": "2",
                    "title": "第二件商品",
                    "category_l1": "家居日用",
                    "category_l2": "保温杯",
                    "image_path": "image.webp",
                    "split": "test",
                },
                {
                    "product_id": "1",
                    "title": "第一件商品",
                    "category_l1": "3C数码",
                    "category_l2": "耳机",
                    "image_path": "image.webp",
                    "split": "train",
                },
            ]
            dataset_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            loaded = load_catalog(dataset_path, root)
            metadata_path = root / "metadata.jsonl"
            write_metadata(metadata_path, loaded)
            metadata = load_metadata(metadata_path)

            self.assertEqual([record["product_id"] for record in metadata], ["2", "1"])
            self.assertNotIn("attributes", metadata[0])

    def test_catalog_rejects_duplicate_product_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "image.webp"
            image_path.write_bytes(b"placeholder")
            record = {
                "product_id": "1",
                "title": "商品",
                "category_l1": "3C数码",
                "category_l2": "耳机",
                "image_path": "image.webp",
                "split": "train",
            }
            dataset_path = root / "products.jsonl"
            line = json.dumps(record, ensure_ascii=False) + "\n"
            dataset_path.write_text(line + line, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "商品编号重复"):
                load_catalog(dataset_path, root)


class RetrievalFeatureTests(unittest.TestCase):
    def test_normalize_rows_returns_unit_vectors(self) -> None:
        normalized = normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2), atol=1e-6)

    def test_embedder_rejects_invalid_batch_before_loading_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            ErnieVilEmbedder(batch_size=0)


if __name__ == "__main__":
    unittest.main()
