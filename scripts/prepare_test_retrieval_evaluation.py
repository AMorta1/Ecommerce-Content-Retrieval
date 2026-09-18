"""Select 50 frozen-test queries and create exhaustive relevance judgments."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_catalog, resolve_project_path  # noqa: E402
from src.retrieval.feature_library import catalog_index_path  # noqa: E402
from src.retrieval.formal_evaluation import (  # noqa: E402
    build_relevance_rows,
    select_balanced_queries,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_test_evaluation.json"
ANNOTATION_FIELDS = [
    "query_id",
    "query_category_l1",
    "query_category_l2",
    "query_text",
    "query_source_title",
    "query_attributes",
    "query_image_path",
    "candidate_order",
    "product_id",
    "title",
    "candidate_attributes",
    "image_path",
    "relevance_grade",
    "review_notes",
]
QUERY_REVIEW_FIELDS = [
    "query_id",
    "product_id",
    "category_l1",
    "category_l2",
    "source_title",
    "source_attributes",
    "query_text",
    "image_path",
    "review_query_ok",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="正式评测配置")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="查询审核通过后生成正式查询、相关性标注表和冻结清单",
    )
    parser.add_argument(
        "--confirm-review",
        action="store_true",
        help="仅确认已有查询审核表并更新冻结清单，不重建标注表",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding=encoding, newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_annotation(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                for field in ("query_attributes", "candidate_attributes"):
                    serialized[field] = json.dumps(
                        serialized[field], ensure_ascii=False, separators=(",", ":")
                    )
                writer.writerow(serialized)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_query_review(
    path: Path,
    queries: list[dict[str, Any]],
    query_texts: dict[str, str],
) -> None:
    """Write 50 natural-language query drafts for human approval."""
    if path.exists():
        raise FileExistsError(f"查询审核表已存在，拒绝覆盖：{path}")
    selected_ids = {query["product_id"] for query in queries}
    if selected_ids != set(query_texts):
        missing = sorted(selected_ids - set(query_texts))
        extra = sorted(set(query_texts) - selected_ids)
        raise ValueError(f"查询文本与选中商品不一致；缺少={missing}，多余={extra}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=QUERY_REVIEW_FIELDS)
            writer.writeheader()
            for query in queries:
                writer.writerow(
                    {
                        "query_id": f"test_{query['product_id']}",
                        "product_id": query["product_id"],
                        "category_l1": query["category_l1"],
                        "category_l2": query["category_l2"],
                        "source_title": query["title"],
                        "source_attributes": json.dumps(
                            query["attributes"], ensure_ascii=False, separators=(",", ":")
                        ),
                        "query_text": query_texts[query["product_id"]],
                        "image_path": query["image_path"],
                        "review_query_ok": "",
                        "review_notes": "",
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_approved_queries(
    path: Path,
    selected_queries: list[dict[str, Any]],
) -> dict[str, str]:
    """Require all 50 selected query texts to be explicitly approved."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    selected_ids = {query["product_id"] for query in selected_queries}
    reviewed_ids = {row["product_id"].strip() for row in rows}
    if len(rows) != len(selected_queries) or reviewed_ids != selected_ids:
        raise ValueError("查询审核表与固定抽样商品不一致。")

    selected_by_id = {query["product_id"]: query for query in selected_queries}
    protected_fields = {
        "query_id": lambda query: f"test_{query['product_id']}",
        "category_l1": lambda query: query["category_l1"],
        "category_l2": lambda query: query["category_l2"],
        "source_title": lambda query: query["title"],
        "source_attributes": lambda query: json.dumps(
            query["attributes"], ensure_ascii=False, separators=(",", ":")
        ),
        "image_path": lambda query: query["image_path"],
    }
    changed_protected_fields = []
    for row in rows:
        product_id = row["product_id"].strip()
        selected = selected_by_id[product_id]
        for field, expected_value in protected_fields.items():
            if row[field].strip() != str(expected_value(selected)).strip():
                changed_protected_fields.append(f"{product_id}:{field}")
    if changed_protected_fields:
        examples = ", ".join(changed_protected_fields[:5])
        raise ValueError(
            "查询审核表中的来源字段不能修改；只允许修改 query_text、review_query_ok "
            f"和 review_notes。异常示例：{examples}"
        )

    not_approved = [row["query_id"] for row in rows if row["review_query_ok"].strip() != "1"]
    if not_approved:
        raise ValueError(f"还有 {len(not_approved)} 条查询未确认，review_query_ok 必须填写1。")
    query_texts = {row["product_id"].strip(): row["query_text"].strip() for row in rows}
    if any(not text for text in query_texts.values()):
        raise ValueError("query_text 不能为空。")
    return query_texts


def ensure_query_review_can_be_confirmed(manifest: dict, metrics_path: Path) -> None:
    if manifest.get("status") == "human_evaluation_completed" or metrics_path.exists():
        raise ValueError(
            "正式检索评测已完成，不能再次运行 --confirm-review；"
            "如需修改查询，请先讨论新的评测版本与人工标注处理方式。"
        )


def main() -> None:
    args = parse_args()
    if args.finalize and args.confirm_review:
        raise ValueError("--finalize 和 --confirm-review 不能同时使用。")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    retrieval_config_path = resolve_project_path(PROJECT_ROOT, config["retrieval_config"])
    retrieval_config = json.loads(retrieval_config_path.read_text(encoding="utf-8"))
    dataset_path = resolve_project_path(PROJECT_ROOT, retrieval_config["dataset_path"])
    records = load_catalog(dataset_path, PROJECT_ROOT)
    test_records = [record for record in records if record["split"] == config["gallery_split"]]
    queries = select_balanced_queries(
        test_records,
        config["query_quota_by_category"],
        int(config["seed"]),
        set(config["excluded_query_product_ids"]),
        config.get("query_replacements", {}),
    )

    query_texts_path = resolve_project_path(PROJECT_ROOT, config["query_texts_path"])
    query_review_path = resolve_project_path(PROJECT_ROOT, config["query_review_path"])
    if args.confirm_review:
        approved_texts = load_approved_queries(query_review_path, queries)
        queries_path = resolve_project_path(PROJECT_ROOT, config["queries_path"])
        manifest_path = resolve_project_path(PROJECT_ROOT, config["manifest_path"])
        with queries_path.open("r", encoding="utf-8") as query_file:
            frozen_queries = [json.loads(line) for line in query_file if line.strip()]
        if len(frozen_queries) != len(queries) or any(
            approved_texts.get(query["product_id"]) != query["query_text"]
            for query in frozen_queries
        ):
            raise ValueError("人工确认的查询文本与已生成的查询清单不一致。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["version"] != config["version"] or manifest["queries_sha256"] != file_sha256(queries_path):
            raise ValueError("冻结清单版本或查询文件哈希不一致。")
        metrics_path = resolve_project_path(PROJECT_ROOT, config["metrics_path"])
        ensure_query_review_can_be_confirmed(manifest, metrics_path)
        manifest["status"] = "awaiting_human_annotation"
        manifest["query_review_sha256"] = file_sha256(query_review_path)
        manifest["query_review_confirmed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print("50 条查询已经人工确认；相关性 AI 初标仍需逐行复核。")
        return
    draft_query_texts = json.loads(query_texts_path.read_text(encoding="utf-8"))
    if not args.finalize:
        write_query_review(query_review_path, queries, draft_query_texts)
        print(
            json.dumps(
                {
                    "status": "awaiting_query_review",
                    "query_count": len(queries),
                    "query_review_path": config["query_review_path"],
                    "instruction": "检查或修改 query_text，并把 review_query_ok 全部填写为1。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    approved_query_texts = load_approved_queries(query_review_path, queries)
    approved_queries = [
        {**record, "query_text": approved_query_texts[record["product_id"]]}
        for record in queries
    ]
    rows = build_relevance_rows(approved_queries, test_records)
    queries_path = resolve_project_path(PROJECT_ROOT, config["queries_path"])
    annotation_path = resolve_project_path(PROJECT_ROOT, config["annotation_path"])
    manifest_path = resolve_project_path(PROJECT_ROOT, config["manifest_path"])
    for path in (queries_path, annotation_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"正式评测文件已存在，拒绝覆盖：{path}")

    query_lines = []
    for record in approved_queries:
        query = {
            "query_id": f"test_{record['product_id']}",
            "product_id": record["product_id"],
            "query_text": record["query_text"],
            "query_image_path": record["image_path"],
            "category_l1": record["category_l1"],
            "category_l2": record["category_l2"],
        }
        query_lines.append(json.dumps(query, ensure_ascii=False))
    atomic_write_text(queries_path, "\n".join(query_lines) + "\n")
    write_annotation(annotation_path, rows)

    features_by_mode = config["catalog_feature_by_query_mode"]
    if set(features_by_mode) != {"text", "image"}:
        raise ValueError("catalog_feature_by_query_mode 必须分别配置 text 和 image。")
    if config["retrieval_strategy"] == "pure_cross_modal" and features_by_mode != {
        "text": "image",
        "image": "text",
    }:
        raise ValueError("纯跨模态策略必须配置为文本查图片、图片查文本。")
    index_paths_by_mode = {
        mode: resolve_project_path(
            PROJECT_ROOT,
            catalog_index_path(retrieval_config, catalog_feature),
        )
        for mode, catalog_feature in features_by_mode.items()
    }
    manifest = {
        "status": "awaiting_human_annotation",
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gallery_split": config["gallery_split"],
        "gallery_size": len(test_records),
        "retrieval_strategy": config["retrieval_strategy"],
        "catalog_feature_by_query_mode": features_by_mode,
        "query_count": len(queries),
        "queries_by_category_l1": dict(sorted(Counter(q["category_l1"] for q in queries).items())),
        "queries_by_category_l2": dict(sorted(Counter(q["category_l2"] for q in queries).items())),
        "excluded_query_product_ids": config["excluded_query_product_ids"],
        "excluded_query_reasons": config["excluded_query_reasons"],
        "query_replacements": config.get("query_replacements", {}),
        "query_product_excluded_from_gallery": True,
        "annotation_scope": "每条查询的同二级品类 test 商品，排除查询商品本身；跨品类视为不相关。",
        "annotation_rows": len(rows),
        "label_values": {
            "2": "高度相关：商品类型和主要属性/用途均兼容",
            "1": "部分相关：商品类型正确，但只满足部分属性",
            "0": "不相关：关键商品类型、属性或用途冲突",
        },
        "queries_sha256": file_sha256(queries_path),
        "query_review_sha256": file_sha256(query_review_path),
        "gallery_sha256": file_sha256(dataset_path.parent / "test.jsonl"),
        "index_sha256_by_query_mode": {
            mode: file_sha256(index_path)
            for mode, index_path in index_paths_by_mode.items()
        },
        "warning": "完成1,200行人工复核前不能计算或宣称正式 test 指标。",
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
