from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from src.data.preprocess import build_dataset, write_json


class PipelineTests(unittest.TestCase):
    def test_pipeline_trace_dedup_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {str(i): {"title": f"product {i}", "label": "A" if i < 12 else "B",
                              "url": f"https://img.alicdn.com/{i}.jpg", "video": "",
                              "pv": f"型号#:#M{i}#;#颜色#:#黑色#;#材质#:#塑料"} for i in range(24)}
            data["100"] = dict(data["0"], url="https://img.alicdn.com/100.jpg")
            data["101"] = dict(data["1"], title="another title")
            data["102"] = dict(data["2"], pv="broken")
            data["103"] = dict(data["3"], label="not in scope")
            source_records = (
                ("full", data),
                ("train", {"0": data["0"]}),
                ("test", {"0": data["0"], "1": data["1"]}),
            )
            for name, records in source_records:
                write_json(root / f"{name}.json", records)
            config = {
                "sources": {
                    name: f"{name}.json" for name in ("full", "train", "test")
                },
                "categories": [
                    {
                        "raw_labels": [name],
                        "category_l1": name,
                        "category_l2": name,
                    }
                    for name in ("A", "B")
                ],
                "max_title_chars": 100,
                "max_pv_chars": 1000,
                "min_attribute_keys": 3,
                "image_hosts": ["img.alicdn.com"],
                "taxonomy_version": "test",
                "dataset_version": "test",
                "seed": 42,
                "per_category": 12,
            }
            write_json(root / "config.json", config)
            with redirect_stdout(StringIO()):
                summary = build_dataset(config, root, root / "output", root / "config.json")
            self.assertEqual(summary["selected"], 24)
            self.assertFalse((root / "output" / "candidates.sqlite").exists())
            self.assertEqual(list(root.glob("ecr-candidates-*")), [])
            self.assertEqual(summary["selected_original_overlap"], 1)
            self.assertEqual(summary["rejected"]["duplicate_content"], 1)
            self.assertEqual(summary["rejected"]["duplicate_image_url"], 1)
            self.assertEqual(summary["rejected"]["malformed_or_insufficient_attributes"], 1)
            splits = {}
            for split in ("train", "validation", "test"):
                split_path = root / "output" / f"{split}.jsonl"
                rows = [
                    json.loads(line)
                    for line in split_path.read_text(encoding="utf-8").splitlines()
                ]
                splits[split] = {row["product_id"] for row in rows}
            self.assertFalse(splits["train"] & splits["test"])
            self.assertFalse(splits["train"] & splits["validation"])
            self.assertFalse(splits["test"] & splits["validation"])
            with self.assertRaises(FileExistsError):
                build_dataset(config, root, root / "output", root / "config.json")
            self.assertEqual(list(root.glob("ecr-candidates-*")), [])
            # 即使原始 JSON 损坏导致处理中断，候选库也应自动清理。
            (root / "full.json").write_text('{"broken":', encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset(config, root, root / "failed_output", root / "config.json")
            self.assertEqual(list(root.glob("ecr-candidates-*")), [])


if __name__ == "__main__":
    unittest.main()
