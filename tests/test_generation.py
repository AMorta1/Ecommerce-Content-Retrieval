import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_generation import (
    build_annotation_rows,
    draft_formal_rows,
    structured_output_success_rate,
    validate_attribution_rows,
    validate_output_report,
)
from scripts.run_rag_validation import (
    load_excluded_product_ids,
    paired_sample_seed,
    select_development_samples,
    verify_p2_holdout_guard,
)
from src.generation.grounding import validate_generation_grounding
from src.generation.evaluation import (
    GenerationJudgment,
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    parse_judgments,
)
from src.generation.qwen import (
    build_messages,
    build_rag_messages,
    mandatory_fact_lines,
    parse_generation_output,
    preflight_chat_prompt,
)
from src.generation.rag import (
    build_fact_units,
    is_placeholder_value,
    select_global_top_k_facts,
    select_rag_v2_context,
    select_top_k_facts,
)


class GenerationConfigTests(unittest.TestCase):
    def test_sampling_parameters_are_active_without_changing_old_baseline(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_dir = project_root / "configs"
        current = json.loads((config_dir / "generation.json").read_text(encoding="utf-8"))
        historical = json.loads(
            (config_dir / "generation_greedy_baseline.json").read_text(encoding="utf-8")
        )

        self.assertTrue(current["do_sample"])
        self.assertEqual(current["temperature"], 0.7)
        self.assertEqual(current["top_p"], 0.9)
        self.assertEqual(current["prompt_version"], "baseline_v2")
        self.assertFalse(historical["do_sample"])
        self.assertEqual(historical["prompt_version"], "baseline_v1")
        self.assertEqual(
            {
                key: value
                for key, value in current.items()
                if key not in ("do_sample", "prompt_version")
            },
            {
                key: value
                for key, value in historical.items()
                if key not in ("do_sample", "prompt_version")
            },
        )

    def test_sampling_evaluation_uses_separate_files_and_same_targets(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        historical = json.loads(
            (config_dir / "generation_evaluation.json").read_text(encoding="utf-8")
        )
        sampling = json.loads(
            (config_dir / "generation_evaluation_sampling_v2.json").read_text(encoding="utf-8")
        )

        self.assertEqual(sampling["version"], "generation_sampling_prompt_v2")
        self.assertEqual(sampling["targets"], historical["targets"])
        for field in ("outputs_path", "annotation_path", "metrics_path"):
            self.assertNotEqual(sampling[field], historical[field])

    def test_rag_validation_keeps_model_and_decoding_parameters(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        baseline = json.loads((config_dir / "generation.json").read_text(encoding="utf-8"))
        for version in ("v1", "v2", "v3", "v4", "v5", "v6"):
            rag = json.loads(
                (config_dir / f"generation_rag_validation_{version}.json").read_text(
                    encoding="utf-8"
                )
            )
            for field in (
                "model_name",
                "device",
                "load_in_4bit",
                "bnb_4bit_quant_type",
                "bnb_4bit_compute_dtype",
                "max_input_tokens",
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "seed",
            ):
                self.assertEqual(rag[field], baseline[field])
            self.assertEqual(rag["baseline_prompt_version"], baseline["prompt_version"])
            self.assertEqual(rag["prompt_version"], f"rag_{version}")
            self.assertFalse(rag["pairing"]["formal_test100"])

    def test_rag_v2_policy_matches_frozen_core_fields_and_denominator_contract(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        policy = json.loads(
            (config_dir / "rag_fact_policy_v4.json").read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (config_dir / "generation_evaluation.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (config_dir / "generation_rag_validation_v7.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(policy["mandatory_core_fields"], evaluation["core_attributes"])
        contract = validation["denominator_contract"]
        self.assertEqual(contract["static_category_field_opportunities"], 1500)
        self.assertFalse(contract["source_missing_is_mandatory_injection_failure"])
        self.assertFalse(contract["official_denominator_modified"])
        self.assertIn("core_attribute_count", contract["official_attribute_hit_denominator"])


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

    def test_v2_prompt_requires_exactly_three_selling_points(self) -> None:
        messages = build_messages(self.generation_input, prompt_version="baseline_v2")

        self.assertIn("恰好有三个字符串", messages[0]["content"])
        self.assertIn("不得添加第四个卖点", messages[1]["content"])

    def test_v1_prompt_remains_available_for_historical_baseline(self) -> None:
        messages = build_messages(self.generation_input, prompt_version="baseline_v1")

        self.assertNotIn("不得添加第四个卖点", messages[1]["content"])

    def test_rag_prompt_uses_only_identity_and_shared_selected_facts(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "category_l2",
                    "normalized_values": ["保温杯"],
                    "quality_status": "eligible",
                }
            ],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "材质",
                    "normalized_values": ["304不锈钢"],
                    "quality_status": "eligible",
                }
            ],
        }

        messages = build_rag_messages(self.generation_input, context)
        prompt = messages[1]["content"]

        self.assertIn("全局共享高置信事实", prompt)
        self.assertIn("304不锈钢", prompt)
        self.assertNotIn("500mL", prompt)
        self.assertIn("不得为了凑满三条", prompt)

    def test_rag_prompt_rejects_noneligible_fact(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "容量",
                    "normalized_values": ["500mL"],
                    "quality_status": "withhold_review",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "eligible"):
            build_rag_messages(self.generation_input, context)

    def test_rag_v2_forbids_identity_inference_and_effect_expansion(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "型号",
                    "normalized_values": ["YOGA2无线鼠标"],
                    "quality_status": "eligible",
                }
            ],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "接口类型",
                    "normalized_values": ["USB"],
                    "quality_status": "eligible",
                }
            ],
        }

        messages = build_rag_messages(
            self.generation_input, context, prompt_version="rag_v2"
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("品牌和型号仅用于标识商品", combined)
        self.assertIn("禁止从品牌名或型号文本", combined)
        self.assertIn("不解释这些事实会带来什么效果", combined)

    def test_rag_v3_uses_neutral_field_style(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "材质",
                    "normalized_values": ["304不锈钢"],
                    "quality_status": "eligible",
                }
            ],
        }

        messages = build_rag_messages(
            self.generation_input, context, prompt_version="rag_v3"
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("字段直述方式", combined)
        self.assertIn("字段：值", combined)
        self.assertIn("只直接列出上下文已有", combined)

    def test_rag_v4_marks_negative_fact_and_forbids_category_scenario_inference(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 3,
            "identity_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "category_l2",
                    "normalized_values": ["键盘"],
                    "quality_status": "eligible",
                }
            ],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "是否机械键盘",
                    "normalized_values": ["否"],
                    "quality_status": "eligible",
                },
                {
                    "product_id": "1",
                    "canonical_field": "是否无线",
                    "normalized_values": ["有线"],
                    "quality_status": "eligible",
                },
            ],
        }

        messages = build_rag_messages(
            self.generation_input, context, prompt_version="rag_v4"
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("否定事实；可以不写成卖点，但禁止生成相反内容", combined)
        self.assertIn("只是分类标签", combined)
        self.assertIn("selling_points 仍须恰好三个字符串", combined)
        self.assertIn("只能用通过质量检查的品类、品牌或型号", combined)

    def test_rag_v5_marks_category_and_places_scenario_check_near_output(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "category_l1",
                    "normalized_values": ["家居日用"],
                    "quality_status": "eligible",
                }
            ],
            "selected_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "材质",
                    "normalized_values": ["塑料"],
                    "quality_status": "eligible",
                }
            ],
        }

        messages = build_rag_messages(
            self.generation_input, context, prompt_version="rag_v5"
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("仅分类标签；禁止推导适用对象、用途或使用场景", combined)
        self.assertIn("错误写法是“适用于家居日用”", combined)
        self.assertIn("只有高置信事实中存在明确的适用场景", combined)

    def test_rag_v6_has_separate_read_only_negative_constraints(self) -> None:
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [
                {
                    "product_id": "1",
                    "canonical_field": "category_l2",
                    "normalized_values": ["键盘"],
                    "quality_status": "eligible",
                }
            ],
            "selected_facts": [
                {
                    "fact_id": "positive",
                    "product_id": "1",
                    "canonical_field": "接口类型",
                    "normalized_values": ["USB"],
                    "quality_status": "eligible",
                }
            ],
            "negative_constraint_facts": [
                {
                    "fact_id": "negative",
                    "product_id": "1",
                    "canonical_field": "是否机械键盘",
                    "normalized_values": ["否"],
                    "quality_status": "eligible",
                }
            ],
        }

        messages = build_rag_messages(
            self.generation_input, context, prompt_version="rag_v6"
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("只读否定约束（不占Top-K，不要求写入文案）", combined)
        self.assertIn("只用于禁止相反内容", combined)
        self.assertIn("品类可以做不增加新事实的直接语义释义", combined)

    def test_rag_v2_context_and_coverage_share_four_zones_but_only_coverage_is_strong(self) -> None:
        def fact(fact_id: str, field: str, value: str) -> dict[str, object]:
            return {
                "fact_id": fact_id,
                "product_id": "1",
                "canonical_field": field,
                "normalized_values": [value],
                "quality_status": "eligible",
            }

        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [fact("identity", "category_l2", "保温杯")],
            "mandatory_core_facts": [fact("mandatory", "容量", "500mL")],
            "supplemental_facts": [fact("supplemental", "是否带吸管", "是")],
            "selected_facts": [fact("supplemental", "是否带吸管", "是")],
            "negative_constraint_facts": [],
            "mandatory_injection_expected_fact_ids": ["mandatory"],
        }

        context_prompt = "\n".join(
            message["content"]
            for message in build_rag_messages(
                self.generation_input, context, prompt_version="rag_v2_context_v1"
            )
        )
        coverage_prompt = "\n".join(
            message["content"]
            for message in build_rag_messages(
                self.generation_input, context, prompt_version="rag_v2_coverage_v1"
            )
        )

        self.assertIn("[mandatory] 容量=500mL", context_prompt)
        self.assertIn("Supplemental Top-3", context_prompt)
        self.assertNotIn("每一条 Mandatory Reliable Core Fact 至少完整出现一次", context_prompt)
        self.assertIn("每一条 Mandatory Reliable Core Fact 至少完整出现一次", coverage_prompt)
        self.assertEqual(mandatory_fact_lines(context), ["[mandatory] 容量=500mL"])

    def test_rag_v2_controlled_factual_prompt_is_fact_only_and_soft_coverage(self) -> None:
        def fact(fact_id: str, field: str, value: str) -> dict[str, object]:
            return {
                "fact_id": fact_id,
                "product_id": "1",
                "canonical_field": field,
                "normalized_values": [value],
                "quality_status": "eligible",
            }

        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [fact("identity", "品牌", "示例品牌")],
            "mandatory_core_facts": [fact("mandatory", "容量", "500mL")],
            "supplemental_facts": [fact("supplemental", "材质", "不锈钢")],
            "negative_constraint_facts": [],
            "mandatory_injection_expected_fact_ids": ["mandatory"],
        }

        prompt = "\n".join(
            message["content"]
            for message in build_rag_messages(
                self.generation_input,
                context,
                prompt_version="rag_v2_controlled_factual_v1",
            )
        )

        self.assertIn("商品事实转写器，不是营销文案创作者", prompt)
        self.assertIn("只能陈述事实是什么，禁止解释因此有什么好处", prompt)
        self.assertIn("品牌和型号只能逐字来自 Identity", prompt)
        self.assertIn("禁止舍入、估算、换算、扩大、缩小", prompt)
        self.assertIn("禁止改写成各种设备、全部设备、广泛兼容", prompt)
        self.assertIn("不得从品类推导具体场所、人群、兼容对象、性能或效果", prompt)
        self.assertIn("三个卖点优先逐条写成“字段：值”", prompt)
        self.assertIn("若某条事实无法自然放入文案，可以不写", prompt)
        self.assertNotIn("每一条 Mandatory Reliable Core Fact 至少完整出现一次", prompt)
        self.assertIn("[mandatory] 容量=500mL", prompt)

    def test_rag_v2_rejects_duplicate_zones_and_mandatory_id_mismatch(self) -> None:
        fact = {
            "fact_id": "same",
            "product_id": "1",
            "canonical_field": "容量",
            "normalized_values": ["500mL"],
            "quality_status": "eligible",
        }
        context = {
            "product_id": "1",
            "top_k": 1,
            "identity_facts": [],
            "mandatory_core_facts": [fact],
            "supplemental_facts": [fact],
            "negative_constraint_facts": [],
            "mandatory_injection_expected_fact_ids": ["same"],
        }

        with self.assertRaisesRegex(ValueError, "不得重复"):
            build_rag_messages(
                self.generation_input, context, prompt_version="rag_v2_context_v1"
            )


class PromptPreflightTests(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self, *, drop_mandatory_when_tokenized: bool = False) -> None:
            self.drop_mandatory_when_tokenized = drop_mandatory_when_tokenized
            self.truncation_values: list[bool] = []

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            truncation: bool | None = None,
            **_: object,
        ) -> object:
            rendered = "\n".join(message["content"] for message in messages)
            if not tokenize:
                return rendered
            self.truncation_values.append(bool(truncation))
            if self.drop_mandatory_when_tokenized:
                rendered = rendered.replace("[f1] 容量=500mL", "")
            return {"input_ids": [self.encode(rendered)]}

        def __call__(self, text: str, *, truncation: bool, **_: object) -> object:
            self.truncation_values.append(bool(truncation))
            return {"input_ids": self.encode(text)}

        @staticmethod
        def decode(input_ids: list[int], **_: object) -> str:
            return "".join(chr(token_id) for token_id in input_ids)

    def test_preflight_disables_truncation_and_verifies_mandatory_tokens(self) -> None:
        tokenizer = self.FakeTokenizer()
        _, metadata = preflight_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "事实：[f1] 容量=500mL"}],
            max_input_tokens=100,
            mandatory_lines=["[f1] 容量=500mL"],
        )

        self.assertEqual(tokenizer.truncation_values, [False, False])
        self.assertTrue(metadata["mandatory_injection_complete"])
        self.assertEqual(metadata["mandatory_verified_count"], 1)

    def test_preflight_fails_on_overflow_or_tokenizer_loss(self) -> None:
        with self.assertRaisesRegex(ValueError, "禁止截断"):
            preflight_chat_prompt(
                self.FakeTokenizer(),
                [{"role": "user", "content": "事实：[f1] 容量=500mL"}],
                max_input_tokens=3,
                mandatory_lines=["[f1] 容量=500mL"],
            )
        with self.assertRaisesRegex(ValueError, "tokenizer 后"):
            preflight_chat_prompt(
                self.FakeTokenizer(drop_mandatory_when_tokenized=True),
                [{"role": "user", "content": "事实：[f1] 容量=500mL"}],
                max_input_tokens=100,
                mandatory_lines=["[f1] 容量=500mL"],
            )


class RagValidationRunnerTests(unittest.TestCase):
    def test_loads_excluded_ids_only_from_validation_report(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "validation_result.json"
            path.write_text(
                json.dumps(
                    {
                        "scope": "validation_only",
                        "results": [{"product_id": "1"}, {"product_id": "2"}],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_excluded_product_ids([path]), {"1", "2"})

    def test_paired_seed_is_independent_of_variant(self) -> None:
        seeds = {
            name: paired_sample_seed(42, 7)
            for name in (
                "V1-control",
                "V2-context",
                "V2-coverage",
                "V2-controlled-factual",
            )
        }

        self.assertEqual(set(seeds.values()), {49})

    def test_p26_config_keeps_frozen_generation_and_retrieval_settings(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        p2 = json.loads(
            (project_root / "configs/generation_rag_validation_v7.json").read_text(
                encoding="utf-8"
            )
        )
        p26 = json.loads(
            (project_root / "configs/generation_rag_validation_v8_p26.json").read_text(
                encoding="utf-8"
            )
        )
        frozen_keys = [
            "model_name",
            "revision",
            "cache_dir",
            "device",
            "load_in_4bit",
            "bnb_4bit_quant_type",
            "bnb_4bit_compute_dtype",
            "max_input_tokens",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "seed",
            "validation_source",
            "audit_path",
            "development_sample_source",
            "protected_holdout",
            "denominator_contract",
            "grounding_validator",
        ]
        for key in frozen_keys:
            self.assertEqual(p26[key], p2[key], key)
        self.assertEqual(
            p26["pairing"]["variants"],
            ["V1-control", "V2-controlled-factual"],
        )
        candidate = p26["comparison_variants"][1]
        self.assertEqual(candidate["prompt_version"], "rag_v2_controlled_factual_v1")
        self.assertEqual(candidate["context_selector"], "rag_v2_four_zone")
        self.assertEqual(candidate["fact_policy_path"], "configs/rag_fact_policy_v4.json")

    def test_frozen_holdout_manifest_and_p2_sample_guard(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs/generation_rag_validation_v7.json").read_text(
                encoding="utf-8"
            )
        )
        holdout_path = project_root / config["protected_holdout"]["path"]
        protected = verify_p2_holdout_guard(
            holdout_path,
            expected_manifest_sha256=config["protected_holdout"]["sha256"],
            validation_path=project_root / config["validation_source"],
            audit_path=project_root / config["audit_path"],
        )

        self.assertEqual(len(protected), 32)
        product_id = next(iter(protected))
        records = [
            {"product_id": product_id, "category_l2": "键盘"},
        ]
        audit = {product_id: {"status": "PASS"}}
        with self.assertRaisesRegex(ValueError, "holdout 重叠"):
            select_development_samples(
                records,
                [product_id],
                audit,
                per_category=1,
                protected_product_ids=protected,
            )


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


class RagFactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cls.policy = json.loads(
            (project_root / "configs/rag_fact_policy_v1.json").read_text(encoding="utf-8")
        )
        cls.policy_v2 = json.loads(
            (project_root / "configs/rag_fact_policy_v2.json").read_text(encoding="utf-8")
        )
        cls.policy_v3 = json.loads(
            (project_root / "configs/rag_fact_policy_v3.json").read_text(encoding="utf-8")
        )
        cls.policy_v4 = json.loads(
            (project_root / "configs/rag_fact_policy_v4.json").read_text(encoding="utf-8")
        )

    def build_facts(
        self,
        *,
        product_id: str = "1",
        status: str = "PASS",
        blocked_fields: str = "",
        issue_code: str = "",
        attributes: dict[str, list[str]] | None = None,
        category_l2: str = "键盘",
        policy: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        record = {
            "product_id": product_id,
            "split": "validation",
            "category_l1": "3C数码",
            "category_l2": category_l2,
            "attributes": attributes
            or {
                "品牌": ["双飞燕"],
                "双飞燕型号": ["KR-6A"],
                "接口类型": ["USB"],
                "是否无线": ["有线"],
                "键数": ["104键"],
                "轴体": ["青轴"],
            },
        }
        audit = {
            "audit_version": "test_audit",
            "scope": "validation",
            "product_id": product_id,
            "status": status,
            "issue_code": issue_code,
            "review_note": "测试记录",
            "blocked_fields": blocked_fields,
        }
        return build_fact_units(
            record,
            audit,
            policy or self.policy,
            dataset_version="test_v1",
            source_path="validation.jsonl",
            source_sha256="abc123",
        )

    def test_identity_is_separate_from_top_k_and_model_alias_is_canonicalized(self) -> None:
        facts = self.build_facts()
        selection = select_top_k_facts(facts, self.policy, task="title", top_k=2)

        identity_fields = [fact["canonical_field"] for fact in selection["identity_facts"]]
        selected_fields = [fact["canonical_field"] for fact in selection["selected_facts"]]
        self.assertEqual(identity_fields, ["category_l1", "category_l2", "品牌", "型号"])
        self.assertEqual(len(selected_fields), 2)
        self.assertNotIn("品牌", selected_fields)
        self.assertNotIn("型号", selected_fields)

    def test_review_withholds_only_flagged_field(self) -> None:
        facts = self.build_facts(status="REVIEW", blocked_fields="接口类型")
        by_field = {fact["canonical_field"]: fact for fact in facts}

        self.assertEqual(by_field["接口类型"]["quality_status"], "withhold_review")
        self.assertEqual(by_field["是否无线"]["quality_status"], "eligible")
        selection = select_top_k_facts(facts, self.policy, task="selling_points", top_k=3)
        self.assertNotIn(
            "接口类型", [fact["canonical_field"] for fact in selection["selected_facts"]]
        )

    def test_identity_conflict_blocks_all_product_facts(self) -> None:
        facts = self.build_facts(
            status="CONFLICT",
            blocked_fields="品牌",
            issue_code="brand_title_attribute_conflict",
        )
        selection = select_top_k_facts(facts, self.policy, task="short_description")

        self.assertTrue(facts)
        self.assertTrue(all(fact["quality_status"] == "blocked_conflict" for fact in facts))
        self.assertEqual(selection["identity_facts"], [])
        self.assertEqual(selection["selected_facts"], [])

    def test_ordinary_conflict_blocks_only_conflicting_field(self) -> None:
        facts = self.build_facts(status="CONFLICT", blocked_fields="键数")
        by_field = {fact["canonical_field"]: fact for fact in facts}

        self.assertEqual(by_field["键数"]["quality_status"], "blocked_conflict")
        self.assertEqual(by_field["接口类型"]["quality_status"], "eligible")
        self.assertEqual(by_field["品牌"]["quality_status"], "eligible")

    def test_multi_value_and_placeholder_values_are_not_selected(self) -> None:
        facts = self.build_facts(
            attributes={
                "品牌": ["其他"],
                "接口类型": ["USB", "PS/2"],
                "是否无线": ["有线"],
            }
        )
        by_field = {fact["canonical_field"]: fact for fact in facts}

        self.assertNotIn("品牌", by_field)
        self.assertEqual(by_field["接口类型"]["quality_status"], "withhold_review")
        selection = select_top_k_facts(facts, self.policy, task="selling_points", top_k=3)
        self.assertEqual(
            [fact["canonical_field"] for fact in selection["selected_facts"]],
            ["是否无线"],
        )

    def test_composite_placeholder_is_filtered_without_hiding_mixed_real_value(self) -> None:
        facts = self.build_facts(
            category_l2="鼠标",
            policy=self.policy_v3,
            attributes={
                "品牌": ["Logitech/罗技"],
                "罗技无线型号": ["G304"],
                "接口类型": ["USB"],
                "光学分辨率": ["其他/other"],
                "工作方式": ["激光"],
            },
        )
        by_field = {fact["canonical_field"]: fact for fact in facts}

        self.assertNotIn("光学分辨率", by_field)
        placeholders = {
            str(value).casefold()
            for value in self.policy_v3["global_value_gates"]["exclude_placeholders"]
        }
        separator = self.policy_v3["global_value_gates"][
            "composite_placeholder_separator_pattern"
        ]
        self.assertTrue(
            is_placeholder_value(
                "其他/other",
                placeholders,
                composite_separator_pattern=separator,
            )
        )
        self.assertFalse(
            is_placeholder_value(
                "other/USB",
                placeholders,
                composite_separator_pattern=separator,
            )
        )

    def test_task_priority_is_deterministic_and_top_k_is_configurable(self) -> None:
        facts = self.build_facts()
        title = select_top_k_facts(facts, self.policy, task="title", top_k=3)
        selling_points = select_top_k_facts(
            facts, self.policy, task="selling_points", top_k=3
        )

        self.assertEqual(
            [fact["canonical_field"] for fact in title["selected_facts"]],
            ["键数", "接口类型", "是否无线"],
        )
        self.assertEqual(
            [fact["canonical_field"] for fact in selling_points["selected_facts"]],
            ["接口类型", "是否无线", "键数"],
        )
        self.assertTrue(
            all("selection_reason" in fact for fact in title["selected_facts"])
        )

    def test_global_top_k_fuses_all_three_task_rankings(self) -> None:
        facts = self.build_facts()
        selection = select_global_top_k_facts(facts, self.policy, top_k=3)

        self.assertEqual(selection["task"], "global_all_content")
        self.assertEqual(len(selection["selected_facts"]), 3)
        self.assertEqual(selection["joint_ranking"]["method"], "reciprocal_rank_fusion")
        for fact in selection["selected_facts"]:
            self.assertEqual(
                set(fact["task_ranks"]), {"title", "selling_points", "short_description"}
            )
            self.assertEqual(
                set(fact["task_score_contributions"]),
                {"title", "selling_points", "short_description"},
            )

    def test_global_selection_separates_eligible_negative_constraints(self) -> None:
        facts = self.build_facts()

        selection = select_global_top_k_facts(
            facts,
            self.policy_v2,
            top_k=3,
            separate_negative_constraints=True,
        )

        selected_fields = {
            fact["canonical_field"] for fact in selection["selected_facts"]
        }
        constraint_fields = {
            fact["canonical_field"]
            for fact in selection["negative_constraint_facts"]
        }
        self.assertNotIn("是否无线", selected_fields)
        self.assertIn("是否无线", constraint_fields)
        self.assertTrue(
            all(
                fact["quality_status"] == "eligible"
                for fact in selection["negative_constraint_facts"]
            )
        )

    def test_negative_constraints_do_not_bypass_quality_gate(self) -> None:
        facts = self.build_facts(status="REVIEW", blocked_fields="是否无线")

        selection = select_global_top_k_facts(
            facts,
            self.policy_v2,
            top_k=3,
            separate_negative_constraints=True,
        )

        self.assertNotIn(
            "是否无线",
            {
                fact["canonical_field"]
                for fact in selection["negative_constraint_facts"]
            },
        )

    def test_selection_rejects_cross_product_facts(self) -> None:
        facts = self.build_facts(product_id="1") + self.build_facts(product_id="2")

        with self.assertRaisesRegex(ValueError, "禁止跨商品"):
            select_top_k_facts(facts, self.policy, task="title")

    def test_missing_audit_status_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "质量审计状态"):
            self.build_facts(status="")

    def test_rag_v2_four_zones_are_disjoint_and_quality_gated(self) -> None:
        facts = self.build_facts(
            policy=self.policy_v4,
            attributes={
                "品牌": ["双飞燕"],
                "双飞燕型号": ["KR-6A"],
                "接口类型": ["USB"],
                "是否无线": ["有线"],
                "是否机械键盘": ["是"],
                "轴体": ["青轴"],
            },
        )
        context = select_rag_v2_context(facts, self.policy_v4, top_k=3)
        zones = [
            context["identity_facts"],
            context["mandatory_core_facts"],
            context["supplemental_facts"],
            context["negative_constraint_facts"],
        ]
        zone_ids = [{fact["fact_id"] for fact in zone} for zone in zones]

        for index, left in enumerate(zone_ids):
            for right in zone_ids[index + 1 :]:
                self.assertFalse(left & right)
        self.assertIn(
            "接口类型",
            {fact["canonical_field"] for fact in context["mandatory_core_facts"]},
        )
        self.assertIn(
            "是否无线",
            {fact["canonical_field"] for fact in context["negative_constraint_facts"]},
        )
        self.assertLessEqual(len(context["supplemental_facts"]), 3)
        self.assertTrue(
            all(
                fact["quality_status"] == "eligible"
                for zone in zones
                for fact in zone
            )
        )

    def test_missing_source_core_field_is_not_mandatory_injection_failure(self) -> None:
        facts = self.build_facts(
            policy=self.policy_v4,
            attributes={"品牌": ["双飞燕"], "接口类型": ["USB"]},
        )
        context = select_rag_v2_context(facts, self.policy_v4, top_k=3)
        mandatory_fields = {
            fact["canonical_field"] for fact in context["mandatory_core_facts"]
        }

        self.assertIn("接口类型", mandatory_fields)
        self.assertNotIn("是否机械键盘", mandatory_fields)
        self.assertEqual(
            len(context["mandatory_injection_expected_fact_ids"]),
            len(context["mandatory_core_facts"]),
        )

    def test_rag_v2_withholds_multivalue_review_conflict_and_invalid_weight(self) -> None:
        multivalue = self.build_facts(
            category_l2="收纳箱",
            policy=self.policy_v4,
            attributes={
                "品牌": ["测试品牌"],
                "材质": ["塑料"],
                "适用空间": ["卧室", "客厅"],
                "净重": ["0kg"],
            },
        )
        by_field = {fact["canonical_field"]: fact for fact in multivalue}

        self.assertEqual(by_field["适用空间"]["quality_status"], "withhold_review")
        self.assertNotIn("净重", by_field)
        reviewed = self.build_facts(
            policy=self.policy_v4,
            status="REVIEW",
            blocked_fields="接口类型",
        )
        conflicted = self.build_facts(
            policy=self.policy_v4,
            status="CONFLICT",
            blocked_fields="品牌",
        )
        self.assertNotIn(
            "接口类型",
            {
                fact["canonical_field"]
                for fact in select_rag_v2_context(reviewed, self.policy_v4)[
                    "mandatory_core_facts"
                ]
            },
        )
        self.assertFalse(
            any(
                select_rag_v2_context(conflicted, self.policy_v4)[key]
                for key in (
                    "identity_facts",
                    "mandatory_core_facts",
                    "supplemental_facts",
                    "negative_constraint_facts",
                )
            )
        )


class GroundingValidatorTests(unittest.TestCase):
    @staticmethod
    def context(
        *,
        category_l2: str,
        selected: list[tuple[str, str]] | None = None,
        negative: list[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        def fact(index: int, field: str, value: str) -> dict[str, object]:
            return {
                "fact_id": f"f{index}:{field}",
                "canonical_field": field,
                "normalized_values": [value],
                "quality_status": "eligible",
            }

        return {
            "identity_facts": [
                fact(0, "category_l1", "家居日用"),
                fact(1, "category_l2", category_l2),
            ],
            "selected_facts": [
                fact(index + 2, field, value)
                for index, (field, value) in enumerate(selected or [])
            ],
            "negative_constraint_facts": [
                fact(index + 20, field, value)
                for index, (field, value) in enumerate(negative or [])
            ],
        }

    @staticmethod
    def output(description: str) -> dict[str, object]:
        return {
            "generated_title": "商品标题",
            "selling_points": ["品牌信息", "型号信息", "品类信息"],
            "short_description": description,
        }

    def test_allows_direct_category_semantics(self) -> None:
        result = validate_generation_grounding(
            self.output("这款收纳箱用于物品收纳。"),
            self.context(category_l2="收纳箱"),
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["classification_counts"].get("supported_paraphrase"), 1
        )

    def test_flags_specific_unsupported_usage_scenario(self) -> None:
        result = validate_generation_grounding(
            self.output("这款收纳箱适合宿舍搬家。"),
            self.context(category_l2="收纳箱"),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["classification_counts"].get("unsupported_usage_scenario"), 1
        )

    def test_allows_generic_mop_cleaning_and_explicit_wearing_paraphrase(self) -> None:
        mop_result = validate_generation_grounding(
            self.output("这款拖把适合家居日常清洁使用。"),
            self.context(category_l2="拖把"),
        )
        headphone_result = validate_generation_grounding(
            self.output("这款耳机适合头戴佩戴。"),
            self.context(category_l2="耳机", selected=[("佩戴方式", "头戴式")]),
        )

        self.assertEqual(mop_result["status"], "PASS")
        self.assertEqual(headphone_result["status"], "PASS")

    def test_flags_unsupported_evaluative_claims(self) -> None:
        context = self.context(
            category_l2="移动电源", selected=[("电池容量", "30000mAh")]
        )
        result = validate_generation_grounding(
            self.output("30000mAh大容量移动电源。"), context
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["classification_counts"].get("unsupported_evaluative_claim"), 1
        )

    def test_flags_quality_and_convenience_without_source_support(self) -> None:
        result = validate_generation_grounding(
            self.output("采用优质材质，方便日常投放。"),
            self.context(category_l2="垃圾桶", selected=[("材质", "不锈钢")]),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["classification_counts"].get("unsupported_evaluative_claim"), 2
        )

    def test_flags_negative_fact_contradiction(self) -> None:
        context = self.context(
            category_l2="键盘", negative=[("是否机械键盘", "否")]
        )
        result = validate_generation_grounding(
            self.output("这是一款机械键盘。"), context
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["classification_counts"].get("grounding_failure"), 1)

    def test_accepts_negative_fact_without_forcing_it_into_copy(self) -> None:
        context = self.context(
            category_l2="键盘", negative=[("是否机械键盘", "否")]
        )
        result = validate_generation_grounding(
            self.output("该商品为键盘。"), context
        )

        self.assertEqual(result["status"], "PASS")

    def test_flags_directional_selected_fact_contradiction(self) -> None:
        context = self.context(
            category_l2="保温杯", selected=[("保温时长", "6小时以下")]
        )
        result = validate_generation_grounding(
            self.output("具有6小时以上的保温能力。"), context
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["classification_counts"].get("grounding_failure"), 1)


class GenerationEvaluationTests(unittest.TestCase):
    @staticmethod
    def draft_fixture() -> tuple[
        list[dict[str, str]],
        dict[str, object],
        dict[str, object],
        list[dict[str, object]],
        dict[str, dict[str, str]],
    ]:
        row = {
            "product_id": "1",
            "core_attribute_count": "2",
            "matched_attribute_count": "",
            "fluency_pass": "",
            "factual_error_count": "",
            "category_style_pass": "",
            "source_fact_quality_count": "",
            "retrieval_error_count": "",
            "grounding_failure_count": "",
            "unsupported_generation_count": "",
            "supported_paraphrase_count": "",
            "review_notes": "",
        }
        output_report = {
            "results": [
                {
                    "product_id": "1",
                    "generation_input": {
                        "attributes": {"接口类型": ["USB"], "键数": ["104键"]}
                    },
                    "parsed_output": {
                        "generated_title": "USB键盘",
                        "selling_points": ["接口为USB", "有线连接", "键盘产品"],
                        "short_description": "这是一款USB接口键盘。",
                    },
                    "rag_context": {
                        "identity_facts": [],
                        "selected_facts": [
                            {"canonical_field": "接口类型", "normalized_values": ["USB"]}
                        ],
                        "negative_constraint_facts": [],
                    },
                }
            ]
        }
        validator_report = {
            "results": [
                {
                    "product_id": "1",
                    "grounding_validation": {
                        "classification_counts": {
                            "unsupported_evaluative_claim": 1,
                            "unsupported_usage_scenario": 1,
                            "grounding_failure": 1,
                            "supported_paraphrase": 2,
                        },
                        "findings": [],
                    },
                }
            ]
        }
        facts = [
            {
                "product_id": "1",
                "canonical_field": "接口类型",
                "quality_status": "eligible",
            },
            {
                "product_id": "1",
                "canonical_field": "键数",
                "quality_status": "eligible",
            },
        ]
        audit = {
            "1": {
                "status": "REVIEW",
                "issue_code": "test_review",
                "review_note": "测试审计记录",
            }
        }
        return [row], output_report, validator_report, facts, audit

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

    def test_formal_rag_annotation_exposes_injected_facts_without_prefilling_labels(
        self,
    ) -> None:
        output_report = {
            "results": [
                {
                    "product_id": "1",
                    "audit_status": "REVIEW",
                    "generation_input": {
                        "category_l1": "3C数码",
                        "category_l2": "键盘",
                        "attributes": {"接口类型": ["USB"], "键数": ["104键"]},
                    },
                    "parsed_output": {
                        "generated_title": "USB键盘",
                        "selling_points": ["接口：USB", "键数：104键", "品类：键盘"],
                        "short_description": "键盘采用USB接口，共104键。",
                    },
                    "raw_output": "",
                    "rag_context": {
                        "identity_facts": [
                            {
                                "canonical_field": "category_l2",
                                "normalized_values": ["键盘"],
                            }
                        ],
                        "selected_facts": [
                            {
                                "canonical_field": "接口类型",
                                "normalized_values": ["USB"],
                            }
                        ],
                        "negative_constraint_facts": [
                            {
                                "canonical_field": "是否无线",
                                "normalized_values": ["有线"],
                            }
                        ],
                    },
                }
            ]
        }

        row = build_annotation_rows(output_report)[0]

        self.assertEqual(row["core_attribute_count"], 2)
        self.assertEqual(row["source_quality_status"], "REVIEW")
        self.assertEqual(row["rag_identity_facts"], "category_l2=键盘")
        self.assertEqual(row["rag_selected_facts"], "接口类型=USB")
        self.assertEqual(row["rag_negative_constraints"], "是否无线=有线")
        self.assertEqual(row["matched_attribute_count"], "")
        self.assertEqual(row["unsupported_generation_count"], "")

    def test_formal_output_success_rate_uses_parsed_count(self) -> None:
        self.assertEqual(
            structured_output_success_rate(
                {"sample_count": 100, "parsed_output_count": 100}
            ),
            1.0,
        )

    def test_error_attribution_must_match_factual_error_count(self) -> None:
        rows = [
            {
                "product_id": "1",
                "factual_error_count": "2",
                "source_fact_quality_count": "1",
                "retrieval_error_count": "0",
                "grounding_failure_count": "1",
                "unsupported_generation_count": "0",
                "supported_paraphrase_count": "0",
            }
        ]
        fields = (
            "source_fact_quality_count",
            "retrieval_error_count",
            "grounding_failure_count",
            "unsupported_generation_count",
            "supported_paraphrase_count",
        )

        with self.assertRaisesRegex(ValueError, "必须等于"):
            validate_attribution_rows(rows, fields)

    def test_formal_output_report_rejects_hash_mismatch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "outputs.json"
            report = {
                "scope": "formal_test100",
                "formal_test100_executed": True,
                "results": [{"product_id": "1"}],
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            config = {
                "expected_outputs_sha256": "0" * 64,
                "expected_scope": "formal_test100",
                "expected_sample_count": 1,
                "require_formal_test100_executed": True,
            }

            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_output_report(config, path, report)

    def test_formal_ai_draft_is_marked_and_uses_traceable_counts(self) -> None:
        rows, outputs, validator, facts, audit = self.draft_fixture()

        drafted = draft_formal_rows(rows, outputs, validator, facts, audit)[0]

        self.assertEqual(drafted["matched_attribute_count"], "1")
        self.assertEqual(drafted["source_fact_quality_count"], "1")
        self.assertEqual(drafted["retrieval_error_count"], "1")
        self.assertEqual(drafted["grounding_failure_count"], "1")
        self.assertEqual(drafted["unsupported_generation_count"], "2")
        self.assertEqual(drafted["supported_paraphrase_count"], "2")
        self.assertEqual(drafted["factual_error_count"], "3")
        self.assertTrue(drafted["review_notes"].startswith("AI初标："))
        self.assertIn("疑似未召回字段=键数", drafted["review_notes"])

    def test_formal_ai_draft_refuses_to_overwrite_existing_review(self) -> None:
        rows, outputs, validator, facts, audit = self.draft_fixture()
        rows[0]["fluency_pass"] = "1"

        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            draft_formal_rows(rows, outputs, validator, facts, audit)


if __name__ == "__main__":
    unittest.main()
