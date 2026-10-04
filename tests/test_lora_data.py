import unittest

from src.generation.lora_data import (
    TASK_TYPES,
    build_dataset,
    build_instruction_text,
    select_reliable_core_attributes,
    source_title_quality_issue,
    validate_dataset,
)


def make_test_config() -> dict:
    return {
        "version": "test_v1",
        "template_version": "template_v1",
        "consistency_review": {
            "confirmed_status": "confirmed",
            "review_status": "review",
        },
        "quality_gates": {
            "exclude_placeholders": ["其他", "other", "如图"],
            "composite_placeholder_separator_pattern": "[/\\|,，;；、]+",
            "identity_absent_values": ["无", "否", "无品牌"],
            "excluded_values_by_field": {"型号": ["保温杯"], "无线技术": ["支持"]},
            "model_reject_patterns": ["^\\d+\\s*(?:m|mah|毫安)$"],
            "numeric_core_fields_requiring_digit": ["容量"],
            "unsafe_attribute_value_terms": ["特价", "礼包"],
            "allowed_multivalue_set_fields": ["功能"],
            "maximum_values_per_set_field": 3,
            "negative_values_not_used_as_active_copy_points": ["无", "否", "不支持"],
            "minimum_active_core_attributes": 3,
            "title_always_remove_terms": ["特价", "旗舰"],
            "title_conditionally_supported_terms": ["家用"],
            "maximum_title_characters": 60,
            "maximum_short_description_attributes": 5,
            "category_title_aliases": {"保温杯": ["保温杯"]},
            "category_excluded_title_terms": {"保温杯": ["杯套"]},
            "mobile_power_capacity_token_pattern": "(?i)(\\d{4,})\\s*(?:mAh|毫安|m\\b)",
            "maximum_distinct_mobile_power_capacity_tokens": 1,
        },
        "targets": {"selling_point_count": 3},
        "split": {
            "namespace": "test",
            "seed": 42,
            "validation_fraction": 0.5,
        },
    }


def product(product_id: str, title: str = "特价旗舰家用甲牌A1 500mL保温杯") -> dict:
    return {
        "product_id": product_id,
        "title": title,
        "category_l1": "家居日用",
        "category_l2": "保温杯",
        "attributes": {
            "品牌": ["甲牌"],
            "型号": ["A1"],
            "容量": ["500mL"],
            "材质": ["304不锈钢"],
            "适用场景": ["家用"],
            "杯子样式": ["直身杯"],
        },
    }


CORE = {
    "保温杯": ["品牌", "型号", "材质", "容量", "杯子样式", "适用场景"]
}


class LoraInstructionDataTests(unittest.TestCase):
    def test_reliable_core_attributes_drop_unsafe_and_unresolved_multivalue(self) -> None:
        record = product("1")
        record["attributes"]["型号"] = ["A1", "A2"]
        record["attributes"]["容量"] = ["特价500mL"]
        selected, actions = select_reliable_core_attributes(
            record, CORE["保温杯"], make_test_config()
        )
        self.assertNotIn("型号", selected)
        self.assertNotIn("容量", selected)
        self.assertIn("withhold_multivalue_field:型号", actions)

    def test_composite_placeholder_and_wrong_field_type_are_removed(self) -> None:
        record = product("1")
        record["attributes"]["品牌"] = ["other/其他"]
        record["attributes"]["容量"] = ["大号保温杯"]
        selected, actions = select_reliable_core_attributes(
            record, CORE["保温杯"], make_test_config()
        )
        self.assertNotIn("品牌", selected)
        self.assertNotIn("容量", selected)
        self.assertTrue(any(action.startswith("drop_placeholder:品牌") for action in actions))
        self.assertTrue(
            any(action.startswith("drop_non_numeric_value:容量") for action in actions)
        )

    def test_source_title_category_and_accessory_boundaries(self) -> None:
        config = make_test_config()
        wrong_category = product("1", "手提袋旅行包")
        accessory = product("2", "保温杯杯套")
        self.assertEqual(
            source_title_quality_issue(wrong_category, config)[0],
            "title_category_not_supported",
        )
        self.assertEqual(
            source_title_quality_issue(accessory, config)[0],
            "accessory_or_product_boundary_title",
        )

    def test_review_and_conflict_products_are_not_accepted(self) -> None:
        records = [product("1"), product("2"), product("3")]
        review = {
            "1": {"review_status": "confirmed", "issue_code": "x", "review_note": "x"},
            "2": {"review_status": "review", "issue_code": "y", "review_note": "y"},
        }
        products, instructions, excluded = build_dataset(
            records, review, CORE, make_test_config()
        )
        self.assertEqual([row["product_id"] for row in products], ["3"])
        self.assertEqual(len(instructions), 3)
        self.assertEqual(
            {row["reason"] for row in excluded},
            {"confirmed_conflict", "review_pending"},
        )

    def test_short_description_uses_natural_boolean_phrasing(self) -> None:
        keyboard_core = {
            "键盘": ["品牌", "型号", "接口类型", "是否无线", "是否机械键盘"]
        }
        record = {
            "product_id": "1",
            "title": "甲牌A1 USB有线机械键盘",
            "category_l1": "3C数码",
            "category_l2": "键盘",
            "attributes": {
                "品牌": ["甲牌"],
                "型号": ["A1"],
                "接口类型": ["USB"],
                "是否无线": ["有线"],
                "是否机械键盘": ["是"],
            },
        }
        config = make_test_config()
        config["quality_gates"]["category_title_aliases"]["键盘"] = ["键盘"]
        config["quality_gates"]["category_excluded_title_terms"]["键盘"] = []
        products, _, _ = build_dataset([record], {}, keyboard_core, config)
        description = products[0]["targets"]["short_description"]
        self.assertIn("连接方式为有线", description)
        self.assertIn("键盘类型为机械键盘", description)
        self.assertNotIn("是否机械键盘为是", description)

    def test_each_product_has_three_tasks_and_original_title_is_not_input(self) -> None:
        source_title = "特价旗舰家用甲牌A1 500mL保温杯"
        products, instructions, excluded = build_dataset(
            [product("1", source_title), product("2", source_title + "二")],
            {},
            CORE,
            make_test_config(),
        )
        validate_dataset(products, instructions, excluded)
        self.assertEqual({row["task_type"] for row in instructions}, set(TASK_TYPES))
        self.assertTrue(all(source_title not in row["instruction"] for row in instructions))
        self.assertTrue(all("source_title" not in row["input"] for row in instructions))
        self.assertEqual(
            len({row["split"] for row in instructions if row["product_id"] == "1"}),
            1,
        )

    def test_targets_only_use_selected_facts_and_selling_points_are_three(self) -> None:
        products, instructions, _ = build_dataset(
            [product("1")], {}, CORE, make_test_config()
        )
        item = products[0]
        title = item["targets"]["title"]
        self.assertNotIn("特价", title)
        self.assertNotIn("旗舰", title)
        self.assertIn("甲牌", title)
        self.assertIn("保温杯", title)
        points = item["targets"]["selling_points"]
        self.assertEqual(len(points), 3)
        for point in points:
            field, value = point.split("：", 1)
            self.assertIn(value, item["used_attributes"][field])
        self.assertEqual(len(instructions), 3)

    def test_instruction_has_only_category_core_attributes_and_task(self) -> None:
        instruction = build_instruction_text(
            "家居日用", "保温杯", "title", {"容量": ["500mL"]}
        )
        self.assertIn("任务类型：title", instruction)
        self.assertIn("可靠核心属性", instruction)
        self.assertNotIn("原标题", instruction)


if __name__ == "__main__":
    unittest.main()
