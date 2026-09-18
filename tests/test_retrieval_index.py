import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.catalog import load_catalog, load_metadata, write_metadata
from src.retrieval.ernie_vil import ErnieVilEmbedder, normalize_rows
from src.retrieval.feature_library import (
    build_product_text,
    catalog_index_path,
    fuse_product_features,
)


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
    def test_formal_baseline_uses_pure_cross_modal_indexes(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "retrieval_test_evaluation.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["retrieval_strategy"], "pure_cross_modal")
        self.assertEqual(
            config["catalog_feature_by_query_mode"],
            {"text": "image", "image": "text"},
        )

    def test_normalize_rows_returns_unit_vectors(self) -> None:
        normalized = normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2), atol=1e-6)

    def test_embedder_rejects_invalid_batch_before_loading_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            ErnieVilEmbedder(batch_size=0)

    def test_product_text_skips_missing_description(self) -> None:
        record = {"product_id": "1", "title": "商品标题", "description": None}
        self.assertEqual(build_product_text(record, ["title", "description"]), "商品标题")

    def test_fused_features_are_normalized(self) -> None:
        text = np.array([[1.0, 0.0]], dtype=np.float32)
        image = np.array([[0.0, 1.0]], dtype=np.float32)
        fused = fuse_product_features(text, image, 0.5, 0.5)
        np.testing.assert_allclose(np.linalg.norm(fused, axis=1), [1.0], atol=1e-6)

    def test_catalog_index_path_validates_mode(self) -> None:
        config = {"index_path": "image.faiss"}
        self.assertEqual(catalog_index_path(config, "image"), "image.faiss")
        with self.assertRaisesRegex(ValueError, "不支持"):
            catalog_index_path(config, "unknown")


if __name__ == "__main__":
    unittest.main()
