import unittest

from src.retrieval.rerank import parse_query_signals, rerank_candidates


POLICY = {
    "category_weights": {"exact_l2": 1.0, "same_l1": 0.7, "cross_l1": 0.4},
    "category_l1_by_l2": {"耳机": "3C数码", "键盘": "3C数码"},
    "category_aliases": {"耳机": ["耳机", "耳塞"], "键盘": ["键盘"]},
    "attribute_concepts": [
        {
            "id": "wireless",
            "query_aliases": ["无线"],
            "support_aliases": ["无线", "蓝牙"],
            "conflict_aliases": ["是否无线有线", "有线连接"],
        },
        {
            "id": "in_ear",
            "query_aliases": ["入耳式"],
            "support_aliases": ["入耳式"],
            "conflict_aliases": ["头戴式"],
        },
    ],
}


def candidate(rank, product_id, score, category, title, attributes):
    return {
        "candidate_rank": str(rank),
        "product_id": product_id,
        "similarity_score": str(score),
        "candidate_category_l1": "3C数码",
        "candidate_category_l2": category,
        "title": title,
        "candidate_attributes": attributes,
        "relevance_grade": "0",
    }


class RetrievalRerankTests(unittest.TestCase):
    def test_query_signals_only_use_query_text(self) -> None:
        signals = parse_query_signals("10000mAh无线入耳式耳机", POLICY)

        self.assertEqual(signals["category_l2"], "耳机")
        self.assertEqual([item["id"] for item in signals["concepts"]], ["wireless", "in_ear"])
        self.assertEqual(signals["numeric_constraints"], ["10000mah"])

    def test_category_scoped_concept_does_not_cross_domains(self) -> None:
        policy = {
            **POLICY,
            "attribute_concepts": [
                {
                    "id": "tablet",
                    "categories": ["键盘"],
                    "query_aliases": ["平板"],
                    "support_aliases": ["平板"],
                    "conflict_aliases": [],
                },
                {
                    "id": "flat_mop",
                    "categories": ["拖把"],
                    "query_aliases": ["平板"],
                    "support_aliases": ["平板"],
                    "conflict_aliases": [],
                },
            ],
        }

        signals = parse_query_signals("适合平板电脑的键盘", policy)

        self.assertEqual([item["id"] for item in signals["concepts"]], ["tablet"])

    def test_rerank_uses_pool_larger_than_ten_and_promotes_matching_candidate(self) -> None:
        candidates = [
            candidate(
                rank,
                str(rank),
                0.4 - rank * 0.001,
                "键盘" if rank == 1 else "耳机",
                "有线头戴式商品" if rank < 11 else "无线蓝牙入耳式耳机",
                {"是否无线": ["有线"]} if rank < 11 else {"佩戴方式": ["入耳式"]},
            )
            for rank in range(1, 13)
        ]

        reranked = rerank_candidates(
            "text",
            candidates,
            POLICY,
            query_text="无线入耳式耳机",
            candidate_pool_depth=12,
            attribute_weight=0.08,
            conflict_weight=0.08,
        )

        self.assertEqual(reranked[0]["product_id"], "11")
        self.assertEqual(reranked[0]["matched_signals"], ["wireless", "in_ear"])
        self.assertGreater(reranked[0]["rerank_score"], reranked[-1]["rerank_score"])

    def test_image_query_is_baseline_passthrough(self) -> None:
        candidates = [
            candidate(rank, str(rank), 0.5 - rank * 0.01, "耳机", "耳机", {})
            for rank in range(1, 12)
        ]

        reranked = rerank_candidates(
            "image",
            candidates,
            POLICY,
            query_text=None,
            candidate_pool_depth=11,
            attribute_weight=1.0,
            conflict_weight=1.0,
        )

        self.assertEqual([row["product_id"] for row in reranked], [str(i) for i in range(1, 12)])
        self.assertTrue(all(row["rerank_score"] is None for row in reranked))

    def test_rejects_top_ten_only_pool(self) -> None:
        candidates = [
            candidate(rank, str(rank), 0.5 - rank * 0.01, "耳机", "耳机", {})
            for rank in range(1, 11)
        ]

        with self.assertRaisesRegex(ValueError, "大于10"):
            rerank_candidates(
                "text",
                candidates,
                POLICY,
                query_text="耳机",
                candidate_pool_depth=10,
                attribute_weight=0.1,
                conflict_weight=0.1,
            )


if __name__ == "__main__":
    unittest.main()
