import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_retrieval_rerank_formal_test import (
    assert_nonnegative_similarities,
    build_gallery_candidate_pool,
    load_baseline_image_passthrough,
)


class FormalRetrievalRerankTests(unittest.TestCase):
    def test_nonnegative_similarity_guard_accepts_observed_direction(self) -> None:
        self.assertEqual(
            assert_nonnegative_similarities([0.24, 0.31], query_id="q1"),
            (0.24, 0.31),
        )

    def test_nonnegative_similarity_guard_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invert the penalty direction"):
            assert_nonnegative_similarities([0.3, -0.01], query_id="q1")

    def test_candidate_pool_filters_split_and_query_product(self) -> None:
        metadata = [
            {"product_id": "q", "split": "test", "title": "query", "category_l1": "A", "category_l2": "a"},
            {"product_id": "train", "split": "train", "title": "train", "category_l1": "A", "category_l2": "a"},
            {"product_id": "1", "split": "test", "title": "one", "category_l1": "A", "category_l2": "a"},
            {"product_id": "2", "split": "test", "title": "two", "category_l1": "A", "category_l2": "a"},
        ]
        catalog = {row["product_id"]: {**row, "attributes": {"field": [row["title"]]}} for row in metadata}
        candidates = build_gallery_candidate_pool(
            np.array([0.9, 0.8, 0.7, 0.6]),
            np.array([0, 1, 2, 3]),
            metadata,
            catalog,
            {"query_id": "q1", "product_id": "q"},
            gallery_split="test",
            candidate_pool_depth=2,
            score_precision=6,
        )
        self.assertEqual([row["product_id"] for row in candidates], ["1", "2"])
        self.assertEqual([row["source_index_rank"] for row in candidates], [3, 4])

    def test_image_passthrough_requires_complete_frozen_top_ten(self) -> None:
        header = "query_mode,query_id,rank,product_id\n"
        rows = "".join(f"image,q1,{rank},p{rank}\n" for rank in range(1, 11))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.csv"
            path.write_text(header + rows, encoding="utf-8-sig")
            grouped = load_baseline_image_passthrough(path, query_ids={"q1"}, cutoff=10)
        self.assertEqual([row["product_id"] for row in grouped["q1"]], [f"p{i}" for i in range(1, 11)])


if __name__ == "__main__":
    unittest.main()
