"""Read-only delivery inventory/statistics; writes only reports/delivery.

Never imports model code, loads weights, calls generation/evaluation, or opens
RAG holdout archives. --capture records an immutable-file baseline; --verify
checks it after approved documentation edits. No cleanup action is implemented.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/delivery"
MUTABLE = {"README.md", "docs/data.md", "docs/generation.md", "docs/retrieval.md", "docs/evaluation.md",
           "docs/delivery_audit.md", "scripts/audit_delivery.py"}
FINAL_SCRIPTS = {"generate", "train_lora", "run_lora_project_test", "run_lora_rag_project_test", "run_rag_formal_test",
                 "build_index", "build_text_features", "search", "evaluate_generation", "evaluate_test_retrieval",
                 "run_retrieval_rerank_formal_test", "verify_lora_v1_best",
                 "prepare_data", "inspect_data", "download_images", "validate_data", "deduplicate_training_data",
                 "build_lora_instruction_data", "build_rag_facts", "validate_rag_outputs", "audit_delivery"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def opaque(path):
    return "holdout" in path.name.casefold()


def files():
    # Git internals are not project delivery content. Inventory artifacts exclude themselves.
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and ".git" not in p.relative_to(ROOT).parts and not p.is_relative_to(OUT))


def read(relative):
    p = ROOT / relative
    if opaque(p):
        raise ValueError("Holdout archive content access is prohibited")
    return json.loads(p.read_text(encoding="utf-8"))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8", errors="replace").strip()


def references():
    result = defaultdict(set)
    def walk(obj, manifest):
        if isinstance(obj, dict):
            if isinstance(obj.get("path"), str) and any(k in obj for k in ("sha256", "weights_sha256")):
                result[obj["path"].replace("\\", "/")].add(manifest)
            for key, value in obj.items():
                if key not in ("code_snapshots", "code_bytes", "code_snapshot"):
                    walk(value, manifest)
        elif isinstance(obj, list):
            for value in obj:
                walk(value, manifest)
    for p in (ROOT / "reports").rglob("*.json"):
        if p.is_relative_to(OUT) or opaque(p) or not any(x in p.name for x in ("manifest", "archive", "checkpoint")):
            continue
        try:
            walk(read(p.relative_to(ROOT)), p.relative_to(ROOT).as_posix())
        except (ValueError, UnicodeError):
            continue
    return result


def classification(relative):
    p = Path(relative); text = p.as_posix(); name = p.name.lower()
    if any(x in p.parts for x in ("__pycache__", ".pytest_cache", ".ruff_cache")) or p.suffix == ".pyc":
        return "DELETE_CANDIDATE", "可再生成的解释器/测试缓存；只建议清理，不执行", "CACHE"
    if opaque(p):
        return "KEEP_REPRODUCIBILITY", "冻结且未执行的generation-output holdout；仅登记文件元数据，不打开内容", "FROZEN"
    if text.startswith("tests/"):
        return "KEEP_FINAL", "回归与保护规则测试（包含工程smoke的测试，不等同于运行模型smoke）", "RECOMMENDED"
    if text.startswith("src/"):
        return "KEEP_FINAL", "数据/生成/检索/评测实现；冻结代码不改动，交付说明标注职责", "RECOMMENDED"
    if text.startswith("scripts/"):
        if "smoke" in name:
            return "ARCHIVE", "工程加载/显存验证入口，不是正式训练或质量评测；保留历史原路径", "SMOKE"
        if name.startswith(("import_", "finalize_", "archive_")):
            return "ARCHIVE", "一次性人工复核导入/归档工具；hash链依赖，不能直接移动或删除", "EXPERIMENT"
        if p.stem in FINAL_SCRIPTS:
            return "KEEP_FINAL", "最终功能或正式实验入口；正式train/test不可当作快速启动命令重跑", "FINAL / RECOMMENDED"
        return "ARCHIVE", "开发/评审/调参/辅助历史入口；可能仍被复现依赖，先逻辑归档", "EXPERIMENT"
    if text.startswith("configs/"):
        if name == "lora_project_validation_v1.json":
            return "KEEP_FINAL", "三任务模板/模型配置为后续正式test冻结继承依赖，不能因validation名称误归过时配置", "FINAL_DEPENDENCY"
        if name in ("lora_project_validation_quick32_v1.json", "lora_rag_v2_development24_v1.json"):
            return "ARCHIVE", "固定开发人审/组合开发配置；保留hash链，但不当最终test或通用推理入口", "EXPERIMENT"
        if any(x in name for x in ("smoke", "candidate", "validation_v", "greedy", "bad_cases", "policy_v1", "policy_v2")):
            return "ARCHIVE", "历史开发或工程候选配置；冻结hash引用需保持原文件和路径", "LEGACY / EXPERIMENT"
        return "KEEP_FINAL", "正式参数/核心字段定义/冻结评测与数据构造配置，状态以最终manifest判定", "FINAL"
    if text.startswith("data/"):
        return "KEEP_REPRODUCIBILITY", "数据/图片/事实单元是复现附件；不提交Git、不移动冻结split", "DATA"
    if text.startswith("artifacts/"):
        if "/lora_v1/" in text or "/features/" in text or "/indexes/" in text:
            return "KEEP_REPRODUCIBILITY", "最终adapter/checkpoint/特征索引；单独交付大文件并保留hash", "FINAL_ARTIFACT"
        return "ARCHIVE", "smoke/acceptance工程权重及状态；不是正式初始化权重，保留历史证据", "SMOKE"
    if text.startswith("reports/"):
        if name == "rag_v2_p2_phase_summary.md":
            return "KEEP_FINAL", "原RAG v2未晋级的最终阶段结论，交付应明确展示", "FINAL"
        if any(x in text for x in ("p6_", "p6_low", "p6_2", "p25", "p26", "development24", "smoke", "rejected", "/history/")):
            return "ARCHIVE", "开发/工程/人工诊断历史材料；只逻辑归档，不能破坏已有manifest", "EXPERIMENT"
        if name in ("phase_report.md", "lora_instruction_data_v1_phase_report.md", "lora_instruction_data_v1_spot_check.md", "lora_instruction_data_v1_samples.csv", "review_quick_start.md"):
            return "ARCHIVE", "阶段过程/抽查/填写操作的历史材料；最终入口用正式数据与技术说明，证据原位保留", "EXPERIMENT"
        if name in ("final_review_report.md", "final_test_analysis.md", "final_training_report.md", "formal_baseline_report.md", "formal_rag_v1_test100_report.md", "formal_rerank_v1_test50_report.md", "rag_v2_p2_phase_summary.md"):
            return "KEEP_FINAL", "交付应展示的正式结论或终止阶段总结", "FINAL"
        if name.endswith("metrics.json") or name in ("human_metrics_unblinded.json", "final_review_manifest.json", "formal_baseline_manifest.json"):
            return "KEEP_FINAL", "正式指标/索引清单；不同评审语境不覆盖彼此", "FINAL"
        return "KEEP_REPRODUCIBILITY", "原始输出、人工标签、环境、hash、manifest及数据审核证据", "FROZEN_EVIDENCE"
    if text.startswith("docs/") or name.startswith("readme") or name.startswith("requirements") or name == ".gitignore":
        return "KEEP_FINAL", "正式交付说明/协议或依赖声明；协议冻结，不随整理修改", "RECOMMENDED"
    return "ARCHIVE", "本地计划/迁移/交接或其他辅助材料，具体交付用途待人工确认；不删除", "LEGACY"


def statistics():
    cfg = read("configs/data.json")
    with (ROOT / "reports/data/census/label_counts.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = {r["raw_label"]: int(r["count"]) for r in csv.DictReader(f)}
    result = {"raw_label_mapped_counts": {c["category_l2"]: sum(labels.get(x, 0) for x in c["raw_labels"]) for c in cfg["categories"]},
              "processing_report": read("reports/data/preprocessing.json"), "week2_summary": read("data/processed/week2_train_v1/summary.json"),
              "p5_final_counts": read("reports/generation/lora/lora_instruction_data_v1_final_manifest.json")["counts"],
              "core_field_definitions": read("configs/generation_evaluation.json")["core_attributes"]}
    splits = {}; ids = {}
    for label, file in {"project_train_v3": "week1_v3/multimodal/train.jsonl", "project_validation": "week1_v3/multimodal/validation.jsonl",
                        "project_test": "week1_v3/multimodal/test.jsonl", "week2_train_v1": "week2_train_v1/train.jsonl",
                        "lora_train": "lora_instruction_data_v1/train.jsonl", "lora_train_dev": "lora_instruction_data_v1/validation.jsonl"}.items():
        p = ROOT / "data/processed" / file
        with p.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ids[label] = {r["product_id"] for r in rows}
        splits[label] = {"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "records": len(rows), "products": len(ids[label]),
                         "category_l1": dict(Counter(r["category_l1"] for r in rows)), "category_l2": dict(Counter(r["category_l2"] for r in rows)),
                         "task_types": dict(Counter(r.get("task_type") for r in rows)) if label.startswith("lora_") else {}}
    result["actual_split_statistics"] = splits
    result["overlap_product_counts"] = {f"{a}__{b}": len(ids[a] & ids[b]) for a,b in (
        ("lora_train", "lora_train_dev"), ("lora_train", "project_validation"), ("lora_train", "project_test"),
        ("lora_train_dev", "project_validation"), ("lora_train_dev", "project_test"), ("project_validation", "project_test"))}
    assert all(n == 0 for n in result["overlap_product_counts"].values())
    assert sum(result["raw_label_mapped_counts"].values()) == 29906
    return result


def runtime_observations():
    """Current local framework evidence, not a historical revision claim."""
    package = Path(sys.executable).parent / "envs/ecommerce-retrieval/Lib/site-packages/paddlenlp"
    cache = ROOT.parent / ".model-cache/paddlenlp/taskflow/PaddlePaddle/ernie_vil-2.0-base-zh"
    paths = [package / "taskflow/multimodal_feature_extraction.py",
             package / "transformers/ernie_vil/modeling.py",
             package / "transformers/ernie_vil/image_processing.py", cache / "preprocessor_config.json"]
    entries = [{"path": str(p), "exists": p.is_file(), "sha256": sha(p) if p.is_file() else None} for p in paths]
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "current installed source/cache only; NOT proof of historical Baseline revision or historical preprocessing hash",
            "feature_position": "Taskflow dynamic get_image_features/get_text_features -> vision_outputs[1]/text_outputs[1]; no project hidden-layer index",
            "taskflow_current_text_max_length": 128, "files": entries,
            "preprocessor_config": json.loads(paths[-1].read_text(encoding="utf-8")) if paths[-1].exists() else "not found; human confirmation required",
            "model_loaded": False, "network_access": False, "historical_weight_revision": "not reliably recorded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.capture == args.verify:
        parser.error("Choose exactly one of --capture/--verify")
    OUT.mkdir(parents=True, exist_ok=True)
    all_files = files(); snapshot = OUT / "protected_files_before.json"
    if args.capture:
        if snapshot.exists():
            raise FileExistsError("Do not overwrite the pre-documentation snapshot")
        baseline = {}
        for p in all_files:
            relative = p.relative_to(ROOT).as_posix()
            if relative in MUTABLE or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            baseline[relative] = {"bytes": p.stat().st_size, "sha256": None if opaque(p) else sha(p), "mtime_ns": p.stat().st_mtime_ns}
        write(snapshot.name, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "files": baseline,
              "git_head": git("rev-parse", "HEAD"), "git_status_before": git("status", "--short"),
              "mutable_document_paths": sorted(MUTABLE), "holdout_content_opened": False})
    else:
        baseline = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
        changed = []
        for name, state in baseline.items():
            p = ROOT / name
            if not p.is_file() or (sha(p) != state["sha256"] if state["sha256"] is not None else (p.stat().st_size != state["bytes"] or p.stat().st_mtime_ns != state["mtime_ns"])):
                changed.append(name)
        if changed:
            raise ValueError(f"Protected files changed: {changed}")
        write("protection_verification.json", {"status": "passed", "protected_files": len(baseline), "changed_files": [],
              "holdout_content_opened": False, "holdout_guard": "size and mtime only; existing parent reports supply state",
              "model_calls": 0, "training_runs": 0, "test_runs": 0, "files_deleted_moved_renamed": 0,
              "git_head_unchanged": git("rev-parse", "HEAD") == json.loads(snapshot.read_text(encoding="utf-8"))["git_head"]})
    tracked = set(git("-c", "core.quotepath=false", "ls-files").splitlines()); refs = references(); inventory = []
    findings = []
    for p in all_files:
        relative = p.relative_to(ROOT).as_posix(); category, reason, role = classification(relative)
        if category == "DELETE_CANDIDATE" and refs[relative]:
            category, reason, role = "KEEP_REPRODUCIBILITY", "虽类似缓存但被冻结manifest引用，不能清理", "FROZEN_EVIDENCE"
        inventory.append({"path": relative, "classification": category, "reason": reason, "entrypoint_role": role,
                          "bytes": p.stat().st_size, "git_tracked": relative in tracked, "frozen_reference_count": len(refs[relative]),
                          "frozen_references": " | ".join(sorted(refs[relative])), "sha256": "" if opaque(p) else sha(p),
                          "action": "NONE; preserve original path if referenced by frozen manifest"})
        if p.suffix == ".py" and p.parts[-2] != "__pycache__":
            text = p.read_text(encoding="utf-8-sig"); tree = ast.parse(text)
            missing = [{"name": n.name, "line": n.lineno} for n in tree.body if isinstance(n, (ast.FunctionDef,ast.ClassDef)) and not ast.get_docstring(n) and not n.name.startswith("_")]
            findings.append({"path": relative, "module_docstring": bool(ast.get_docstring(tree)), "public_top_level_without_docstring": missing,
                             "local_absolute_path_literals": sorted({n.value for n in ast.walk(tree)
                                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and re.match(r"^[A-Za-z]:[\\/]", n.value)}),
                             "unclear_filename_candidate": bool(re.fullmatch(r"(?:test\d+|tmp|new|run\d+|final_v\d+_new)", p.stem))})
    with (OUT / "repository_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    write("data_statistics.json", statistics())
    write("runtime_observations.json", runtime_observations())
    write("code_readability_findings.json", {"method": "static AST/literal scan, no execution", "files": findings})
    totals = {k: {"files": sum(r["classification"]==k for r in inventory), "bytes": sum(r["bytes"] for r in inventory if r["classification"]==k)}
              for k in ("KEEP_FINAL","KEEP_REPRODUCIBILITY","ARCHIVE","DELETE_CANDIDATE")}
    write("inventory_summary.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "all repository files including ignored; excludes .git internals and reports/delivery self-artifacts",
          "file_count": len(inventory), "classifications": totals, "git_head": git("rev-parse","HEAD"), "holdout_content_opened": False,
          "no_file_actions_executed": True, "documentation_targets": sorted(MUTABLE), "inventory_sha256": sha(OUT / "repository_inventory.csv")})
    print(json.dumps({"mode": "capture" if args.capture else "verify", "files": len(inventory), "classifications": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
