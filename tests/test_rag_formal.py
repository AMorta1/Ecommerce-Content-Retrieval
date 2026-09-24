import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_rag_formal_test import (
    build_formal_inputs,
    file_sha256,
    preflight,
    qwen_runtime_config,
)
from src.generation.rag import load_fact_policy


class FormalPreflightTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        def write(relative: str, content: str) -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            return path

        prompt_path = write(
            "src/generation/qwen.py",
            "def build_rag_messages(value):\n    return value\n",
        )
        policy_path = write("configs/policy.json", '{"version":"rag_fact_policy_v3"}\n')
        test100_path = write("data/test100.jsonl", "this is deliberately not JSONL\n")
        audit_path = write("reports/audit.csv", "scope,product_id,status\n")
        baseline_path = write("reports/baseline.json", "{}\n")
        runner_path = write("scripts/run_rag_formal_test.py", "# frozen runner\n")
        snapshot_path = root / "model-cache/snapshot"
        snapshot_path.mkdir(parents=True)

        prompt_source = prompt_path.read_text(encoding="utf-8").strip()
        prompt_sha = hashlib.sha256(prompt_source.encode("utf-8")).hexdigest()
        config = {
            "version": "rag_v1_formal",
            "status": "frozen_pending_single_test100_execution",
            "formal_test100_executed": False,
            "model": {
                "name": "model",
                "revision": "revision",
                "observed_snapshot_path": str(snapshot_path),
            },
            "prompt": {
                "version": "rag_v6",
                "implementation": "src/generation/qwen.py::build_rag_messages",
                "implementation_sha256": prompt_sha,
            },
            "rag": {
                "policy_version": "rag_fact_policy_v3",
                "policy_path": "configs/policy.json",
                "policy_sha256": file_sha256(policy_path),
                "quality_audit_path": "reports/audit.csv",
                "quality_audit_sha256": file_sha256(audit_path),
                "test100_fact_units_path": "data/formal/facts.jsonl",
                "top_k": 3,
                "joint_ranking": {
                    "method": "reciprocal_rank_fusion",
                    "rrf_k": 60,
                },
                "negative_constraints": {"enabled": True},
            },
            "frozen_test100": {
                "path": "data/test100.jsonl",
                "sha256": file_sha256(test100_path),
            },
            "baseline_comparison": {
                "manifest_path": "reports/baseline.json",
                "manifest_sha256": file_sha256(baseline_path),
            },
            "formal_outputs": {
                "overwrite_existing": False,
                "raw_outputs_path": "reports/formal/outputs.json",
                "validator_report_path": "reports/formal/validator.json",
                "human_annotations_path": "reports/formal/review.csv",
                "metrics_path": "reports/formal/metrics.json",
                "report_path": "reports/formal/report.md",
            },
            "execution": {
                "entrypoint": "scripts/run_rag_formal_test.py",
                "seed_rule": "base_seed_plus_stable_sample_index",
                "local_files_only": True,
            },
            "generation_parameters": {"seed": 42},
            "validator": {"version": "validator_v1"},
        }
        config_path = write(
            "configs/formal.json",
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        )
        manifest = {
            "status": "frozen_pending_single_test100_execution",
            "formal_test100_executed": False,
            "formal_config": {"sha256": file_sha256(config_path)},
            "rag_policy": {"sha256": file_sha256(policy_path)},
            "prompt": {"sha256": prompt_sha},
            "frozen_test100": {"sha256": file_sha256(test100_path)},
            "quality_audit": {"sha256": file_sha256(audit_path)},
            "model": {"name": "model", "revision": "revision"},
            "generation_parameters": {"seed": 42},
            "validator": {"version": "validator_v1"},
            "code_version": {
                "formal_runtime_artifacts": [
                    {
                        "path": "scripts/run_rag_formal_test.py",
                        "sha256": file_sha256(runner_path),
                    }
                ]
            },
        }
        manifest_path = write(
            "reports/formal_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        return config_path, manifest_path, config

    def test_preflight_does_not_parse_test100_or_load_model(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, manifest_path, _ = self.make_fixture(root)

            result = preflight(config_path, manifest_path, project_root=root)

            self.assertEqual(result["config"]["version"], "rag_v1_formal")
            self.assertFalse(result["config"]["formal_test100_executed"])

    def test_preflight_rejects_existing_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, manifest_path, config = self.make_fixture(root)
            output = root / config["formal_outputs"]["raw_outputs_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                preflight(config_path, manifest_path, project_root=root)

    def test_preflight_rejects_changed_frozen_config(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, manifest_path, _ = self.make_fixture(root)
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "formal 配置 SHA256 不匹配"):
                preflight(config_path, manifest_path, project_root=root)


class FormalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.policy = load_fact_policy(root / "configs/rag_fact_policy_v3.json")

    @staticmethod
    def record(product_id: str) -> dict:
        return {
            "product_id": product_id,
            "split": "test",
            "category_l1": "3C数码",
            "category_l2": "键盘",
            "title": f"商品 {product_id}",
            "attributes": {
                "品牌": ["双飞燕"],
                "双飞燕型号": ["KR-6A"],
                "接口类型": ["USB"],
                "是否无线": ["有线"],
                "键数": ["104键"],
            },
            "generation_input": {
                "category_l1": "3C数码",
                "category_l2": "键盘",
                "attributes": {
                    "接口类型": ["USB"],
                    "是否无线": ["有线"],
                    "键数": ["104键"],
                },
            },
        }

    @staticmethod
    def audit(product_id: str, *, status: str, blocked_fields: str = "") -> dict:
        return {
            "audit_version": "test",
            "scope": "test100",
            "product_id": product_id,
            "status": status,
            "issue_code": "test",
            "review_note": "test",
            "blocked_fields": blocked_fields,
        }

    def test_build_inputs_retains_conflict_record_and_applies_quality_gate(self) -> None:
        records = [self.record("1"), self.record("2")]
        audit = {
            "1": self.audit("1", status="PASS"),
            "2": self.audit("2", status="CONFLICT", blocked_fields="品牌"),
        }

        facts, prepared = build_formal_inputs(
            records,
            audit,
            self.policy,
            dataset_version="test",
            source_path="synthetic.jsonl",
            source_sha256="abc",
            top_k=3,
        )

        self.assertEqual([row["product_id"] for row in prepared], ["1", "2"])
        self.assertEqual(records[0]["split"], "test")
        self.assertTrue(prepared[0]["rag_context"]["selected_facts"])
        self.assertTrue(prepared[0]["rag_context"]["negative_constraint_facts"])
        self.assertEqual(prepared[1]["rag_context"]["identity_facts"], [])
        self.assertEqual(prepared[1]["rag_context"]["selected_facts"], [])
        self.assertTrue(
            all(
                fact["quality_status"] == "blocked_conflict"
                for fact in facts
                if fact["product_id"] == "2"
            )
        )

    def test_runtime_config_preserves_frozen_model_and_decoding_values(self) -> None:
        formal = {
            "model": {
                "name": "model",
                "revision": "revision",
                "cache_dir": "cache",
                "device": "cuda",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            "generation_parameters": {
                "max_input_tokens": 2048,
                "max_new_tokens": 320,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
            },
            "prompt": {"version": "rag_v6"},
        }

        runtime = qwen_runtime_config(formal)

        self.assertEqual(runtime["model_name"], "model")
        self.assertEqual(runtime["revision"], "revision")
        self.assertEqual(runtime["prompt_version"], "rag_v6")
        self.assertEqual(runtime["temperature"], 0.7)
        self.assertEqual(runtime["seed"], 42)


if __name__ == "__main__":
    unittest.main()
