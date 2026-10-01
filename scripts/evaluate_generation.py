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
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.evaluation import (  # noqa: E402
    MANUAL_FIELDS,
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    load_annotation_rows,
    parse_judgments,
)
from src.generation.rag import load_quality_audit  # noqa: E402


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
    "source_quality_status",
    "rag_identity_facts",
    "rag_selected_facts",
    "rag_negative_constraints",
    "matched_attribute_count",
    "fluency_pass",
    "factual_error_count",
    "category_style_pass",
    "source_fact_quality_count",
    "retrieval_error_count",
    "grounding_failure_count",
    "unsupported_generation_count",
    "supported_paraphrase_count",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="生成评测配置 JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="从模型输出创建空白人工评测表")
    mode.add_argument(
        "--draft-rag-formal",
        action="store_true",
        help="为 formal RAG 人工表写入可识别、不可直接作为正式指标的 AI 初标。",
    )
    parser.add_argument(
        "--allow-assistant-draft",
        action="store_true",
        help="允许计算 AI 初标的临时指标；结果不能作为正式人工评测",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


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


def flatten_rag_facts(facts: list[dict[str, Any]]) -> str:
    return " | ".join(
        f"{fact['canonical_field']}={','.join(str(value) for value in fact['normalized_values'])}"
        for fact in facts
    )


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
        rag_context = result.get("rag_context", {})
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
                "source_quality_status": result.get("audit_status", ""),
                "rag_identity_facts": flatten_rag_facts(
                    rag_context.get("identity_facts", [])
                ),
                "rag_selected_facts": flatten_rag_facts(
                    rag_context.get("selected_facts", [])
                ),
                "rag_negative_constraints": flatten_rag_facts(
                    rag_context.get("negative_constraint_facts", [])
                ),
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
        )
    return rows


def validate_output_report(
    config: dict[str, Any], outputs_path: Path, output_report: dict[str, Any]
) -> None:
    expected_hash = config.get("expected_outputs_sha256")
    if expected_hash and file_sha256(outputs_path) != expected_hash:
        raise ValueError("模型输出 SHA256 与正式评测配置不一致。")
    expected_scope = config.get("expected_scope")
    if expected_scope and output_report.get("scope") != expected_scope:
        raise ValueError("模型输出 scope 与正式评测配置不一致。")
    if config.get("require_formal_test100_executed") is True:
        if output_report.get("formal_test100_executed") is not True:
            raise ValueError("模型输出没有确认执行正式 test100。")
    results = output_report.get("results")
    if not isinstance(results, list):
        raise ValueError("模型输出缺少 results 数组。")
    expected_count = config.get("expected_sample_count")
    if expected_count is not None and len(results) != int(expected_count):
        raise ValueError("模型输出样本数与正式评测配置不一致。")
    product_ids = [str(result.get("product_id", "")) for result in results]
    if any(not product_id for product_id in product_ids) or len(set(product_ids)) != len(
        product_ids
    ):
        raise ValueError("模型输出包含空或重复的 product_id。")


def prepare_annotation(config: dict[str, Any]) -> None:
    outputs_path = resolve_project_path(config["outputs_path"])
    annotation_path = resolve_project_path(config["annotation_path"])
    if annotation_path.exists():
        raise FileExistsError("人工评测表已经存在，不会覆盖可能已经填写的内容。")
    output_report = load_json(outputs_path)
    validate_output_report(config, outputs_path, output_report)
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


def structured_output_success_rate(output_report: dict[str, Any]) -> float:
    if "structured_output_success_rate" in output_report:
        return float(output_report["structured_output_success_rate"])
    sample_count = int(output_report["sample_count"])
    if sample_count < 1:
        raise ValueError("模型输出 sample_count 必须大于0。")
    return int(output_report["parsed_output_count"]) / sample_count


def validate_attribution_rows(
    rows: list[dict[str, str]], attribution_fields: tuple[str, ...]
) -> None:
    for row in rows:
        product_id = row["product_id"].strip()
        counts = {}
        for field in attribution_fields:
            try:
                counts[field] = int(row[field])
            except ValueError as error:
                raise ValueError(f"商品 {product_id} 的 {field} 必须填写整数。") from error
            if counts[field] < 0:
                raise ValueError(f"商品 {product_id} 的 {field} 不能小于0。")
        model_error_count = (
            counts.get("grounding_failure_count", 0)
            + counts.get("unsupported_generation_count", 0)
        )
        if int(row["factual_error_count"]) != model_error_count:
            raise ValueError(
                f"商品 {product_id} 的 factual_error_count 必须等于 "
                "grounding_failure_count + unsupported_generation_count。"
            )


def normalize_match_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def canonical_review_field(field: str) -> str:
    if field == "型号" or field.endswith("型号"):
        return "型号"
    return field


def estimate_matched_attribute_count(
    generation_input: dict[str, Any], parsed_output: dict[str, Any]
) -> int:
    output_text = normalize_match_text(
        " ".join(
            [
                str(parsed_output["generated_title"]),
                *(str(point) for point in parsed_output["selling_points"]),
                str(parsed_output["short_description"]),
            ]
        )
    )
    matched = 0
    for raw_values in generation_input["attributes"].values():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if any(
            normalized and normalized in output_text
            for normalized in (normalize_match_text(value) for value in values)
        ):
            matched += 1
    return matched


def draft_formal_rows(
    rows: list[dict[str, str]],
    output_report: dict[str, Any],
    validator_report: dict[str, Any],
    facts: list[dict[str, Any]],
    audit: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output_by_id = {str(result["product_id"]): result for result in output_report["results"]}
    validator_by_id = {
        str(result["product_id"]): result for result in validator_report["results"]
    }
    facts_by_id: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        facts_by_id.setdefault(str(fact["product_id"]), []).append(fact)
    expected_ids = set(output_by_id)
    if set(validator_by_id) != expected_ids or set(facts_by_id) != expected_ids:
        raise ValueError("formal 输出、validator 和事实单元的商品集合不一致。")

    draft_fields = (
        "matched_attribute_count",
        "fluency_pass",
        "factual_error_count",
        "category_style_pass",
        "source_fact_quality_count",
        "retrieval_error_count",
        "grounding_failure_count",
        "unsupported_generation_count",
        "supported_paraphrase_count",
    )
    drafted = []
    for row in rows:
        product_id = row["product_id"].strip()
        if product_id not in output_by_id or product_id not in audit:
            raise ValueError(f"人工表商品 {product_id} 缺少 formal 证据。")
        if any(row[field].strip() for field in draft_fields) or row["review_notes"].strip():
            raise ValueError(f"商品 {product_id} 已有人工或初标内容，拒绝覆盖。")
        result = output_by_id[product_id]
        parsed = result["parsed_output"]
        if parsed is None:
            matched = 0
            fluency = 0
            category_style = 0
        else:
            matched = estimate_matched_attribute_count(result["generation_input"], parsed)
            fluency = 1
            category_style = 1

        grounding = validator_by_id[product_id]["grounding_validation"]
        classification_counts = grounding.get("classification_counts", {})
        grounding_failure = int(classification_counts.get("grounding_failure", 0))
        unsupported = int(classification_counts.get("unsupported_usage_scenario", 0)) + int(
            classification_counts.get("unsupported_evaluative_claim", 0)
        )
        supported = int(classification_counts.get("supported_paraphrase", 0))

        context = result["rag_context"]
        context_fields = {
            canonical_review_field(str(fact["canonical_field"]))
            for key in ("identity_facts", "selected_facts", "negative_constraint_facts")
            for fact in context.get(key, [])
        }
        eligible_fields = {
            canonical_review_field(str(fact["canonical_field"]))
            for fact in facts_by_id[product_id]
            if fact.get("quality_status") == "eligible"
        }
        core_fields = {
            canonical_review_field(str(field))
            for field in result["generation_input"]["attributes"]
        }
        missing_retrieval_fields = sorted((core_fields & eligible_fields) - context_fields)
        audit_row = audit[product_id]
        source_quality_count = int(audit_row["status"] != "PASS")

        finding_summary = []
        for finding in grounding.get("findings", []):
            finding_summary.append(
                f"{finding['classification']}:{finding.get('location', '')}:"
                f"{finding.get('text_span', '')}"
            )
        notes = [
            "AI初标：仅作为人工复核草稿，不得直接生成正式指标",
            f"属性命中按规范化原值保守匹配={matched}/{row['core_attribute_count']}",
            f"流畅/品类风格暂按结构完整初判={fluency}/{category_style}",
        ]
        if audit_row["status"] != "PASS":
            notes.append(
                f"源审计={audit_row['status']}:{audit_row.get('issue_code', '')}:"
                f"{audit_row.get('review_note', '')}"
            )
        if missing_retrieval_fields:
            notes.append(f"疑似未召回字段={','.join(missing_retrieval_fields)}")
        if finding_summary:
            notes.append("validator候选=" + "；".join(finding_summary))
        notes.append("请人工逐项核对命中、事实边界、重复错误计数与归因")
        drafted.append(
            {
                **row,
                "matched_attribute_count": str(matched),
                "fluency_pass": str(fluency),
                "factual_error_count": str(grounding_failure + unsupported),
                "category_style_pass": str(category_style),
                "source_fact_quality_count": str(source_quality_count),
                "retrieval_error_count": str(len(missing_retrieval_fields)),
                "grounding_failure_count": str(grounding_failure),
                "unsupported_generation_count": str(unsupported),
                "supported_paraphrase_count": str(supported),
                "review_notes": "；".join(notes),
            }
        )
    return drafted


def write_annotation_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def draft_rag_formal_annotation(config: dict[str, Any]) -> None:
    annotation_path = resolve_project_path(config["annotation_path"])
    outputs_path = resolve_project_path(config["outputs_path"])
    validator_path = resolve_project_path(config["validator_path"])
    facts_path = resolve_project_path(config["fact_units_path"])
    audit_path = resolve_project_path(config["quality_audit_path"])
    if not annotation_path.is_file():
        raise FileNotFoundError("请先创建空白 formal 人工评测表。")
    for label, path, expected in (
        ("formal 输出", outputs_path, config["expected_outputs_sha256"]),
        ("validator", validator_path, config["expected_validator_sha256"]),
        ("事实单元", facts_path, config["expected_fact_units_sha256"]),
        ("质量审计", audit_path, config["expected_quality_audit_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} SHA256 与初标配置不一致。")
    output_report = load_json(outputs_path)
    validate_output_report(config, outputs_path, output_report)
    rows = load_annotation_rows(annotation_path)
    drafted = draft_formal_rows(
        rows,
        output_report,
        load_json(validator_path),
        load_jsonl(facts_path),
        load_quality_audit(audit_path, "test100"),
    )
    write_annotation_rows(annotation_path, drafted)
    print(f"已写入 {len(drafted)} 条 AI 初标：{annotation_path}")


def evaluate(config: dict[str, Any], allow_assistant_draft: bool = False) -> None:
    annotation_path = resolve_project_path(config["annotation_path"])
    rows = load_annotation_rows(annotation_path)
    attribution_fields = tuple(config.get("required_attribution_fields", ()))
    required_fields = (*MANUAL_FIELDS, *attribution_fields)
    progress = annotation_progress(rows, required_fields=required_fields)
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

    if attribution_fields:
        validate_attribution_rows(rows, attribution_fields)
    judgments = parse_judgments(rows)
    metrics = evaluate_judgments(judgments)
    outputs_path = resolve_project_path(config["outputs_path"])
    output_report = load_json(outputs_path)
    validate_output_report(config, outputs_path, output_report)
    generation_times = [float(result["generation_seconds"]) for result in output_report["results"]]
    metrics["structured_output_success_rate"] = structured_output_success_rate(output_report)
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
        "version": config.get("version", "generation_baseline_v1"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation_file_sha256": file_sha256(annotation_path),
        "source_outputs_sha256": file_sha256(outputs_path),
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
    elif args.draft_rag_formal:
        draft_rag_formal_annotation(config)
    else:
        evaluate(config, allow_assistant_draft=args.allow_assistant_draft)


if __name__ == "__main__":
    main()
