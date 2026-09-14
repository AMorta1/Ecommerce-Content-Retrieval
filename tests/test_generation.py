import json
import unittest

from src.generation.evaluation import (
    GenerationJudgment,
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    parse_judgments,
)
from src.generation.qwen import build_messages, parse_generation_output


class GenerationPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generation_input = {
            "category_l1": "家居日用",
            "category_l2": "保温杯",
            "attributes": {"材质": ["304不锈钢"], "容量": ["500mL"]},
        }

    def test_prompt_contains_only_generation_facts(self) -> None:
        messages = build_messages(self.generation_input)
        prompt = messages[1]["content"]

        self.assertIn("304不锈钢", prompt)
        self.assertIn("500mL", prompt)
        self.assertIn("自然生活化", prompt)
        self.assertNotIn("原始商品标题", prompt)

    def test_rejects_target_title_in_generation_input(self) -> None:
        leaked_input = dict(self.generation_input, title="不应该进入提示词的标题")

        with self.assertRaisesRegex(ValueError, "不能包含目标标题"):
            build_messages(leaked_input)


class GenerationOutputTests(unittest.TestCase):
    def test_parses_json_code_block(self) -> None:
        expected = {
            "generated_title": "304不锈钢500mL保温杯",
            "selling_points": ["304不锈钢材质", "500mL容量", "适合日常使用"],
            "short_description": "一款适合日常使用的保温杯。",
        }
        raw_output = f"```json\n{json.dumps(expected, ensure_ascii=False)}\n```"

        self.assertEqual(parse_generation_output(raw_output), expected)

    def test_rejects_wrong_number_of_selling_points(self) -> None:
        raw_output = json.dumps(
            {
                "generated_title": "保温杯",
                "selling_points": ["一个卖点"],
                "short_description": "描述",
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "三个非空字符串"):
            parse_generation_output(raw_output)


class GenerationEvaluationTests(unittest.TestCase):
    def test_zero_values_are_completed_annotations(self) -> None:
        rows = [
            {
                "matched_attribute_count": "0",
                "fluency_pass": "0",
                "factual_error_count": "0",
                "category_style_pass": "0",
            }
        ]

        self.assertEqual(annotation_progress(rows)["completed_rows"], 1)

    def test_assistant_draft_rows_are_counted_from_review_notes(self) -> None:
        rows = [
            {"review_notes": "AI初标：未见明显问题"},
            {"review_notes": "AI初标：存在事实错误"},
            {"review_notes": "人工备注"},
            {"review_notes": ""},
        ]

        self.assertEqual(count_assistant_draft_rows(rows), 2)

    def test_metrics_use_attribute_level_denominator(self) -> None:
        judgments = [
            GenerationJudgment("1", 4, 3, 1, 0, 1),
            GenerationJudgment("2", 6, 3, 0, 2, 1),
        ]

        metrics = evaluate_judgments(judgments)

        self.assertEqual(metrics["core_attribute_hit_rate"], 0.6)
        self.assertEqual(metrics["fluency_pass_rate"], 0.5)
        self.assertEqual(metrics["factual_error_sample_rate"], 0.5)
        self.assertEqual(metrics["category_style_pass_rate"], 1.0)

    def test_rejects_matched_count_above_attribute_count(self) -> None:
        rows = [
            {
                "product_id": "1",
                "core_attribute_count": "2",
                "matched_attribute_count": "3",
                "fluency_pass": "1",
                "factual_error_count": "0",
                "category_style_pass": "1",
            }
        ]

        with self.assertRaisesRegex(ValueError, "属性命中数"):
            parse_judgments(rows)


if __name__ == "__main__":
    unittest.main()
