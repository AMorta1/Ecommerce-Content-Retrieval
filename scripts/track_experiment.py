"""将生成或检索评测结果登记到统一的多版本实验对比表。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "reports" / "experiment_log.csv"

METRIC_FIELDS = (
    "core_attribute_hit_rate",
    "fluency_pass_rate",
    "factual_error_sample_rate",
    "average_factual_errors_per_sample",
    "category_style_pass_rate",
    "structured_output_success_rate",
    "precision_at_10",
    "pooled_recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "average_latency_seconds",
)
DELTA_FIELDS = (
    "delta_core_attribute_hit_rate",
    "delta_fluency_pass_rate",
    "factual_error_rate_reduction",
    "delta_structured_output_success_rate",
    "delta_precision_at_10",
    "delta_pooled_recall_at_10",
    "delta_mrr_at_10",
    "delta_ndcg_at_10",
    "delta_average_latency_seconds",
)
LOG_FIELDS = (
    "module",
    "version",
    "method",
    "model",
    "evaluation_set_id",
    "created_at_utc",
    "metrics_path",
    "metrics_sha256",
    "manifest_path",
    "manifest_sha256",
    "config_paths",
    "parameters_json",
    *METRIC_FIELDS,
    "comparison_baseline_version",
    "comparison_status",
    *DELTA_FIELDS,
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("generation", "retrieval"), required=True)
    parser.add_argument("--method", required=True, help="实验方法，例如 baseline、lora、rag、rerank")
    parser.add_argument("--version", help="实验版本；默认读取指标文件中的 version")
    parser.add_argument("--metrics", type=Path, required=True, help="评测指标 JSON")
    parser.add_argument("--manifest", type=Path, required=True, help="固定评测集清单 JSON")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="本次实验配置 JSON；可重复传入",
    )
    parser.add_argument("--notes", default="", help="简短实验说明")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="实验汇总 CSV")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def require_human_evaluation(module: str, report: dict[str, Any]) -> None:
    allowed_status = {
        "generation": "human_evaluation_completed",
        "retrieval": "pooled_evaluation_completed",
    }[module]
    if report.get("status") != allowed_status:
        raise ValueError(
            f"{module} 指标状态必须为 {allowed_status!r}，当前为 {report.get('status')!r}。"
        )
    provenance = report.get("annotation_provenance", {})
    if int(provenance.get("assistant_draft_rows", 0)) != 0:
        raise ValueError("指标中仍包含 AI 初标，不能登记为正式实验。")


def evaluation_set_id(module: str, manifest: dict[str, Any]) -> str:
    if module == "generation":
        samples_sha256 = manifest.get("samples_sha256")
        if not samples_sha256:
            raise ValueError("生成评测清单缺少 samples_sha256。")
        return f"generation:{samples_sha256}"

    query_sha256 = manifest.get("query_file_sha256")
    gallery_split = manifest.get("gallery_split")
    gallery_size = manifest.get("gallery_size")
    if not query_sha256 or gallery_split is None or gallery_size is None:
        raise ValueError("检索评测清单缺少查询或候选库标识。")
    return f"retrieval:{query_sha256}:{gallery_split}:{gallery_size}"


def extract_metrics(
    module: str, report: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, str]:
    result = {field: "" for field in METRIC_FIELDS}
    metrics = report["metrics"]
    if module == "generation":
        for field in METRIC_FIELDS[:6]:
            result[field] = format_number(metrics.get(field))
        result["average_latency_seconds"] = format_number(
            metrics.get("generation_time_seconds", {}).get("average")
        )
        return result

    macro = metrics["macro_average"]
    for field in ("precision_at_10", "pooled_recall_at_10", "mrr_at_10", "ndcg_at_10"):
        result[field] = format_number(macro.get(field))
    timings = manifest.get("timings_seconds", {})
    query_count = int(manifest.get("query_count", 0))
    if query_count:
        total_query_time = float(timings.get("query_encoding", 0)) + float(
            timings.get("faiss_full_search", 0)
        )
        result["average_latency_seconds"] = format_number(total_query_time / query_count)
    return result


def find_model(configs: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    for config in configs:
        model = config.get("model_name")
        if model:
            return str(model)
    return str(manifest.get("model", ""))


def build_record(
    *,
    module: str,
    method: str,
    version_override: str | None,
    metrics_path: Path,
    manifest_path: Path,
    config_paths: list[Path],
    notes: str,
) -> dict[str, str]:
    report = load_json(metrics_path)
    manifest = load_json(manifest_path)
    configs = [load_json(path) for path in config_paths]
    require_human_evaluation(module, report)
    version = version_override or str(report.get("version", "")).strip()
    if not version:
        raise ValueError("指标文件没有 version，请通过 --version 指定实验版本。")

    config_snapshot = {
        display_path(path): config for path, config in zip(config_paths, configs)
    }
    record = {field: "" for field in LOG_FIELDS}
    record.update(
        {
            "module": module,
            "version": version,
            "method": method.strip().casefold(),
            "model": find_model(configs, manifest),
            "evaluation_set_id": evaluation_set_id(module, manifest),
            "created_at_utc": str(report.get("created_at_utc", "")),
            "metrics_path": display_path(metrics_path),
            "metrics_sha256": file_sha256(metrics_path),
            "manifest_path": display_path(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "config_paths": " | ".join(display_path(path) for path in config_paths),
            "parameters_json": json.dumps(
                config_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "notes": notes.strip(),
        }
    )
    record.update(extract_metrics(module, report, manifest))
    return record


def read_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(LOG_FIELDS):
            raise ValueError(f"实验表列结构与当前脚本不一致：{path}")
        return list(reader)


def metric_delta(current: dict[str, str], baseline: dict[str, str], field: str) -> str:
    if not current[field] or not baseline[field]:
        return ""
    return format_number(float(current[field]) - float(baseline[field]))


def refresh_comparisons(rows: list[dict[str, str]]) -> None:
    delta_mapping = {
        "delta_core_attribute_hit_rate": "core_attribute_hit_rate",
        "delta_fluency_pass_rate": "fluency_pass_rate",
        "delta_structured_output_success_rate": "structured_output_success_rate",
        "delta_precision_at_10": "precision_at_10",
        "delta_pooled_recall_at_10": "pooled_recall_at_10",
        "delta_mrr_at_10": "mrr_at_10",
        "delta_ndcg_at_10": "ndcg_at_10",
        "delta_average_latency_seconds": "average_latency_seconds",
    }
    for row in rows:
        for field in DELTA_FIELDS:
            row[field] = ""
        row["comparison_baseline_version"] = ""
        if row["method"] == "baseline":
            row["comparison_status"] = "baseline"
            continue

        baselines = [
            candidate
            for candidate in rows
            if candidate["module"] == row["module"]
            and candidate["evaluation_set_id"] == row["evaluation_set_id"]
            and candidate["method"] == "baseline"
        ]
        if len(baselines) != 1:
            row["comparison_status"] = (
                "no_matching_baseline" if not baselines else "multiple_matching_baselines"
            )
            continue
        baseline = baselines[0]
        row["comparison_baseline_version"] = baseline["version"]
        row["comparison_status"] = "compared"
        for delta_field, metric_field in delta_mapping.items():
            row[delta_field] = metric_delta(row, baseline, metric_field)
        if row["factual_error_sample_rate"] and baseline["factual_error_sample_rate"]:
            row["factual_error_rate_reduction"] = format_number(
                float(baseline["factual_error_sample_rate"])
                - float(row["factual_error_sample_rate"])
            )


def upsert_record(rows: list[dict[str, str]], record: dict[str, str]) -> str:
    key = (record["module"], record["version"])
    for index, row in enumerate(rows):
        if (row["module"], row["version"]) == key:
            rows[index] = record
            return "updated"
    rows.append(record)
    return "added"


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    metrics_path = resolve_path(args.metrics)
    manifest_path = resolve_path(args.manifest)
    config_paths = [resolve_path(path) for path in args.config]
    log_path = resolve_path(args.log)
    record = build_record(
        module=args.module,
        method=args.method,
        version_override=args.version,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        config_paths=config_paths,
        notes=args.notes,
    )
    rows = read_log(log_path)
    action = upsert_record(rows, record)
    refresh_comparisons(rows)
    write_log(log_path, rows)
    print(
        json.dumps(
            {
                "status": action,
                "module": record["module"],
                "version": record["version"],
                "experiment_count": len(rows),
                "log_path": display_path(log_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
        print(f"实验登记失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
