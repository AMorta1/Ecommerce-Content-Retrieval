from pathlib import Path
import tempfile
import unittest

from scripts.prepare_test_retrieval_evaluation import ensure_query_review_can_be_confirmed

from src.retrieval.evaluation import (
    RelevanceJudgment,
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    parse_completed_judgments,
)
from src.retrieval.formal_evaluation import (
    build_relevance_rows,
    evaluate_complete_rankings,
    parse_complete_relevance,
    select_balanced_queries,
    validate_query_review,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_annotation_progress_does_not_treat_zero_as_empty(self) -> None:
        rows = [{"relevance_grade": "0"}, {"relevance_grade": ""}, {"relevance_grade": "2"}]

        result = annotation_progress(rows)

        self.assertEqual(result["completed_rows"], 2)
        self.assertEqual(result["remaining_rows"], 1)

    def test_assistant_draft_rows_are_counted_from_review_notes(self) -> None:
        rows = [
            {"review_notes": "AI初标，需人工复核"},
            {"review_notes": "AI初标：仅满足部分条件"},
            {"review_notes": "人工备注"},
            {"review_notes": ""},
        ]

        self.assertEqual(count_assistant_draft_rows(rows), 2)

    def test_incomplete_annotation_is_rejected(self) -> None:
        rows = [
            {"query_id": "q1", "candidate_rank": "1", "product_id": "p1", "relevance_grade": ""}
        ]

        with self.assertRaisesRegex(ValueError, "尚未完成"):
            parse_completed_judgments(rows)

    def test_macro_metrics_use_graded_and_binary_relevance(self) -> None:
        judgments = [
            RelevanceJudgment("q1", 1, "p1", 2),
            RelevanceJudgment("q1", 2, "p2", 0),
            RelevanceJudgment("q1", 3, "p3", 1),
            RelevanceJudgment("q1", 4, "p4", 0),
        ]

        result = evaluate_judgments(judgments, cutoff=2)

        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["macro_average"]["precision_at_2"], 0.5)
        self.assertEqual(result["macro_average"]["pooled_recall_at_2"], 0.5)
        self.assertEqual(result["macro_average"]["mrr_at_2"], 1.0)
        self.assertGreater(result["macro_average"]["ndcg_at_2"], 0.7)

    def test_query_without_relevant_candidate_is_rejected(self) -> None:
        judgments = [
            RelevanceJudgment("q1", 1, "p1", 0),
            RelevanceJudgment("q1", 2, "p2", 0),
        ]

        with self.assertRaisesRegex(ValueError, "没有相关商品"):
            evaluate_judgments(judgments, cutoff=2)

    def test_non_contiguous_candidate_ranks_are_rejected(self) -> None:
        judgments = [
            RelevanceJudgment("q1", 1, "p1", 1),
            RelevanceJudgment("q1", 3, "p2", 0),
        ]

        with self.assertRaisesRegex(ValueError, "连续"):
            evaluate_judgments(judgments, cutoff=2)


class FormalRetrievalEvaluationTests(unittest.TestCase):
    def test_confirm_review_rejects_completed_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            with self.assertRaisesRegex(ValueError, "正式检索评测已完成"):
                ensure_query_review_can_be_confirmed(
                    {"status": "human_evaluation_completed"}, metrics_path
                )

            metrics_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "正式检索评测已完成"):
                ensure_query_review_can_be_confirmed(
                    {"status": "awaiting_human_annotation"}, metrics_path
                )

            metrics_path.unlink()
            ensure_query_review_can_be_confirmed(
                {"status": "awaiting_human_annotation"}, metrics_path
            )

    def test_balanced_query_selection_is_deterministic(self) -> None:
        records = [
            {
                "product_id": str(index),
                "category_l2": "耳机" if index < 5 else "键盘",
            }
            for index in range(10)
        ]
        quotas = {"耳机": 2, "键盘": 2}
        first = select_balanced_queries(records, quotas, 42, set())
        second = select_balanced_queries(list(reversed(records)), quotas, 42, set())
        self.assertEqual(
            [record["product_id"] for record in first],
            [record["product_id"] for record in second],
        )

    def test_query_replacement_preserves_category_and_position(self) -> None:
        records = [
            {"product_id": str(index), "category_l2": "鼠标" if index < 4 else "收纳箱"}
            for index in range(8)
        ]
        original = select_balanced_queries(records, {"鼠标": 1, "收纳箱": 1}, 42, set())
        old_id = original[0]["product_id"]
        new_id = next(record["product_id"] for record in records[:4] if record["product_id"] != old_id)

        replaced = select_balanced_queries(
            records, {"鼠标": 1, "收纳箱": 1}, 42, set(), {old_id: new_id}
        )

        self.assertEqual([record["product_id"] for record in replaced],
                         [new_id, original[1]["product_id"]])

    def test_query_replacement_rejects_different_category(self) -> None:
        records = [
            {"product_id": "mouse", "category_l2": "鼠标"},
            {"product_id": "box", "category_l2": "收纳箱"},
        ]
        with self.assertRaisesRegex(ValueError, "二级品类不一致"):
            select_balanced_queries(
                records, {"鼠标": 1}, 42, set(), {"mouse": "box"}
            )

    def test_pending_replacement_query_cannot_be_formally_evaluated(self) -> None:
        review_rows = [{"query_id": "test_new", "query_text": "无线静音游戏鼠标", "review_query_ok": ""}]
        queries = [{"query_id": "test_new", "query_text": "无线静音游戏鼠标"}]
        with self.assertRaisesRegex(ValueError, "未完成人工确认"):
            validate_query_review(review_rows, queries)

    def test_relevance_rows_exclude_query_product(self) -> None:
        gallery = [
            {
                "product_id": str(index),
                "category_l1": "3C数码",
                "category_l2": "耳机",
                "title": f"商品{index}",
                "query_text": "无线蓝牙耳机",
                "attributes": {},
                "image_path": f"{index}.jpg",
            }
            for index in range(3)
        ]
        rows = build_relevance_rows([gallery[0]], gallery)
        self.assertEqual({row["product_id"] for row in rows}, {"1", "2"})
        self.assertEqual({row["query_text"] for row in rows}, {"无线蓝牙耳机"})
        self.assertEqual({row["query_source_title"] for row in rows}, {"商品0"})

    def test_complete_recall_uses_all_judged_relevant_products(self) -> None:
        rows = [
            {"query_id": "q1", "product_id": "p1", "candidate_order": "1", "relevance_grade": "2"},
            {"query_id": "q1", "product_id": "p2", "candidate_order": "2", "relevance_grade": "1"},
            {"query_id": "q1", "product_id": "p3", "candidate_order": "3", "relevance_grade": "0"},
        ]
        relevance = parse_complete_relevance(rows, 1)
        result = evaluate_complete_rankings({"q1": ["p1", "p3"]}, relevance, 2, 1)
        self.assertEqual(result["macro_average"]["precision_at_2"], 0.5)
        self.assertEqual(result["macro_average"]["recall_at_2"], 0.5)

if __name__ == "__main__":
    unittest.main()
