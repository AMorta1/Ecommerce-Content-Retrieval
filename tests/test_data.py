import json
from pathlib import Path
import tempfile
import unittest

from src.data.pv_parser import parse_pv
from src.data.streaming import iter_products
from src.data.preprocess import (
    assign_splits,
    canonical_url,
    content_fingerprint,
    is_laptop_replacement_keyboard,
)


class StreamingTests(unittest.TestCase):
    def read(self, text, chunk=7, limit=4096):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            path.write_text(text, encoding="utf-8")
            return list(iter_products(path, chunk_size=chunk, max_record_chars=limit))

    def test_nested_escaped_and_multibyte(self):
        data = {"001": {"title": '中文{}\\"', "list": [1, {"a": True}]}, "2": {"pv": "颜色#:#蓝"}}
        self.assertEqual(self.read(json.dumps(data, ensure_ascii=False)), list(data.items()))

    def test_formatting_independent(self):
        data = {"1": {"title": "a"}, "2": {"title": "b"}}
        for indent in (None, 2, 4):
            self.assertEqual(self.read(json.dumps(data, indent=indent)), list(data.items()))

    def test_empty_bom_and_duplicate_ids(self):
        self.assertEqual(self.read('\ufeff{} \n'), [])
        self.assertEqual(len(self.read('{"1":{},"1":{}}')), 2)

    def test_invalid_input_fails(self):
        for text in ('{"1":{', '{"1":{},}', '{} {}', '[]', '{"1":null}', '{"1":{} "2":{}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_record_limit(self):
        with self.assertRaises(ValueError):
            self.read(json.dumps({"1": {"title": "a" * 200}}), limit=100)


class AttributeTests(unittest.TestCase):
    def test_multivalue_dedup_and_normalization(self):
        attrs, issues = parse_pv("颜色#:# 黑色 #;#颜色#:#白色#;#颜色#:#黑色#;#型号#:#Ａ１")
        self.assertEqual(attrs, {"颜色": ["黑色", "白色"], "型号": ["A1"]})
        self.assertEqual(issues, [])

    def test_malformed_parts_are_reported(self):
        attrs, issues = parse_pv("错误#;#品牌#:#品牌甲#;##:#空键#;#空值#:#")
        self.assertEqual(attrs, {"品牌": ["品牌甲"]})
        self.assertEqual(len(issues), 3)

    def test_empty(self):
        self.assertEqual(parse_pv("")[0], {})
        self.assertTrue(parse_pv(None)[1])


class DatasetTests(unittest.TestCase):
    def test_laptop_replacement_keyboard_is_distinguished_from_external_keyboard(self):
        self.assertTrue(
            is_laptop_replacement_keyboard(
                "华硕A542笔记本键盘更换",
                {"接口类型": ["PS/2"]},
            )
        )
        self.assertTrue(
            is_laptop_replacement_keyboard(
                "lenovo ThinkPad笔记本电脑键盘全新原装正品",
                {"接口类型": ["蓝牙"]},
            )
        )
        self.assertTrue(
            is_laptop_replacement_keyboard(
                "冠泽 ASUS华硕N53J K55D B53S P55V X53A K54HR键盘",
                {"接口类型": ["黑色(PS/2);白色(USB)"]},
            )
        )
        self.assertFalse(
            is_laptop_replacement_keyboard(
                "戴尔有线键盘笔记本电脑键盘台式外接USB键盘",
                {"接口类型": ["USB"]},
            )
        )
        self.assertFalse(
            is_laptop_replacement_keyboard(
                "USB有线键盘适用笔记本台式电脑",
                {"接口类型": ["USB"]},
            )
        )
        self.assertFalse(
            is_laptop_replacement_keyboard(
                "CHERRY樱桃G80有线游戏机械键盘",
                {"接口类型": ["PS/2", "USB"]},
            )
        )

    def test_stratified_split_is_order_independent(self):
        rows = [{"product_id": str(i), "category_l2": "a" if i < 20 else "b"} for i in range(40)]
        a = assign_splits([dict(row) for row in rows], 42)
        b = assign_splits([dict(row) for row in reversed(rows)], 42)
        self.assertEqual({r["product_id"]: r["split"] for r in a}, {r["product_id"]: r["split"] for r in b})
        for category in ("a", "b"):
            self.assertEqual(sum(r["category_l2"] == category and r["split"] == "train" for r in a), 16)
            self.assertEqual(sum(r["category_l2"] == category and r["split"] == "validation" for r in a), 2)
            self.assertEqual(sum(r["category_l2"] == category and r["split"] == "test" for r in a), 2)

    def test_fingerprint_ignores_attribute_order(self):
        a = content_fingerprint("title", {"颜色": ["黑", "白"], "型号": ["a"]})
        b = content_fingerprint("title", {"型号": ["a"], "颜色": ["白", "黑"]})
        self.assertEqual(a, b)

    def test_url_aliases_and_query_parameters(self):
        self.assertEqual(
            canonical_url("http://img.alicdn.com/a.jpg"),
            canonical_url("https://img.alicdn.com/a.jpg"),
        )
        self.assertNotEqual(
            canonical_url("http://img.alicdn.com/a.jpg?v=1"),
            canonical_url("http://img.alicdn.com/a.jpg?v=2"),
        )


if __name__ == "__main__":
    unittest.main()
