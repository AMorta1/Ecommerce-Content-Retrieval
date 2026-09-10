"""准备文案人工评测表，或根据已完成人工标注计算基线指标。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.evaluation import (  # noqa: E402
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    load_annotation_rows,
    parse_judgments,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generation_evaluation.json"
ANNOTATION_FIELDS = [
    "product_id",
    "category_l1",
    "category_l2",
    "attribute_checklist",
    "core_attribute_count",
    "generated_title",
    "selling_points",
    "short_description",
    "format_valid",
    "auto_exact_matched_count",
    "matched_attribute_count",
    "fluency_pass",
    "factual_error_count",
    "category_style_pass",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="生成评测配置 JSON")
    parser.add_argument("--prepare", action="store_true", help="从模型输出创建空白人工评测表")
    parser.add_argument(
        "--allow-assistant-draft",
        action="store_true",
        help="允许计算 AI 初标的临时指标；结果不能作为正式人工评测",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_attributes(attributes: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    return [(name, [str(value) for value in values]) for name, values in attributes.items()]


def build_annotation_rows(output_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in output_report["results"]:
        generation_input = result["generation_input"]
        attributes = flatten_attributes(generation_input["attributes"])
        format_valid = result["parsed_output"] is not None
        generated = result["parsed_output"]
        if generated is None:
            try:
                generated = json.loads(result["raw_output"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"商品 {result['product_id']} 的原始输出无法展示到人工评测表。"
                ) from error
        if not all(field in generated for field in ("generated_title", "selling_points", "short_description")):
            raise ValueError(f"商品 {result['product_id']} 的原始输出缺少文案字段。")
        generated_text = "\n".join(
            [generated["generated_title"], *generated["selling_points"], generated["short_description"]]
        ).casefold()
        exact_matched = sum(
            any(value.casefold() in generated_text for value in values)
            for _, values in attributes
        )
        rows.append(
            {
                "product_id": result["product_id"],
                "category_l1": generation_input["category_l1"],
                "category_l2": generation_input["category_l2"],
                "attribute_checklist": " | ".join(
                    f"{name}={','.join(values)}" for name, values in attributes
                ),
                "core_attribute_count": len(attributes),
                "generated_title": generated["generated_title"],
                "selling_points": " | ".join(generated["selling_points"]),
                "short_description": generated["short_description"],
                "format_valid": int(format_valid),
                "auto_exact_matched_count": exact_matched,
                "matched_attribute_count": "",
                "fluency_pass": "",
                "factual_error_count": "",
                "category_style_pass": "",
                "review_notes": "",
            }
        )
    return rows


def prepare_annotation(config: dict[str, Any]) -> None:
    outputs_path = resolve_project_path(config["outputs_path"])
    annotation_path = resolve_project_path(config["annotation_path"])
    if annotation_path.exists():
        raise FileExistsError("人工评测表已经存在，不会覆盖可能已经填写的内容。")
    output_report = load_json(outputs_path)
    rows = build_annotation_rows(output_report)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = annotation_path.with_name(annotation_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, annotation_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"已创建 {len(rows)} 条人工评测表：{annotation_path}")


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, int(len(ordered) * 0.95 + 0.999999) - 1)
    return ordered[index]


def evaluate(config: dict[str, Any], allow_assistant_draft: bool = False) -> None:
    annotation_path = resolve_project_path(config["annotation_path"])
    rows = load_annotation_rows(annotation_path)
    progress = annotation_progress(rows)
    if progress["remaining_rows"]:
        print(json.dumps({"status": "annotation_incomplete", **progress}, ensure_ascii=False, indent=2))
        raise ValueError("人工标注尚未完成；填完四个人工评分列后再计算正式指标。")

    assistant_draft_rows = count_assistant_draft_rows(rows)
    if assistant_draft_rows and not allow_assistant_draft:
        print(
            json.dumps(
                {
                    "status": "assistant_draft_awaiting_human_review",
                    "assistant_draft_rows": assistant_draft_rows,
                    "human_reviewed_rows": len(rows) - assistant_draft_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise ValueError("检测到 AI 初标；人工复核并移除 AI初标 标记后才能生成正式指标。")

    judgments = parse_judgments(rows)
    metrics = evaluate_judgments(judgments)
    output_report = load_json(resolve_project_path(config["outputs_path"]))
    generation_times = [float(result["generation_seconds"]) for result in output_report["results"]]
    metrics["structured_output_success_rate"] = output_report["structured_output_success_rate"]
    metrics["generation_time_seconds"] = {
        "average": statistics.fmean(generation_times),
        "p95": percentile_95(generation_times),
        "maximum": max(generation_times),
    }
    targets = config["targets"]
    report = {
        "status": (
            "assistant_draft_evaluation_completed"
            if assistant_draft_rows
            else "human_evaluation_completed"
        ),
        "version": "generation_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation_file_sha256": file_sha256(annotation_path),
        "annotation_progress": progress,
        "annotation_provenance": {
            "human_reviewed_rows": len(rows) - assistant_draft_rows,
            "assistant_draft_rows": assistant_draft_rows,
        },
        "metrics": metrics,
        "prd_target_reference": {
            "core_attribute_hit_rate": {
                "target": targets["core_attribute_hit_rate"],
                "observed": metrics["core_attribute_hit_rate"],
                "passed": metrics["core_attribute_hit_rate"] >= targets["core_attribute_hit_rate"],
            },
            "fluency_pass_rate": {
                "target": targets["fluency_pass_rate"],
                "observed": metrics["fluency_pass_rate"],
                "passed": metrics["fluency_pass_rate"] >= targets["fluency_pass_rate"],
            },
            "average_generation_seconds": {
                "target_maximum": targets["average_generation_seconds"],
                "observed": metrics["generation_time_seconds"]["average"],
                "passed": metrics["generation_time_seconds"]["average"]
                <= targets["average_generation_seconds"],
            },
        },
        "metric_definition": {
            "core_attribute_hit_rate": "所有样本人工确认的命中属性数之和 / 输入核心属性数之和",
            "fluency_pass_rate": "标题、卖点、短详情整体通顺的样本数 / 总样本数",
            "factual_error_sample_rate": "至少含一项输入无法支持或与输入矛盾事实的样本数 / 总样本数",
        },
        "warning": (
            "该结果含 AI 初标，只能用于辅助检查，不能作为正式人工评测或 PRD 达标结论。"
            if assistant_draft_rows
            else "该结果来自已完成的人工复核标注。"
        ),
    }
    metrics_path = resolve_project_path(config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metrics_path.with_name(metrics_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(report, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, metrics_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    if args.prepare:
        prepare_annotation(config)
    else:
        evaluate(config, allow_assistant_draft=args.allow_assistant_draft)


if __name__ == "__main__":
    main()
