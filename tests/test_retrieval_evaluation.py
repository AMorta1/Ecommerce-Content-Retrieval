import unittest

from src.retrieval.evaluation import (
    RelevanceJudgment,
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    parse_completed_judgments,
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


if __name__ == "__main__":
    unittest.main()
