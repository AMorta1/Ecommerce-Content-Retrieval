"""固化正式 test50 的 20 条 3C 定性 Bad Case，不计算 rerank test 指标。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_3c_bad_cases_v1.json"
OUTPUT_FIELDS = [
    "case_id",
    "query_mode",
    "query_id",
    "query_product_id",
    "category_l2",
    "query_text",
    "candidate_product_id",
    "original_rank",
    "similarity_score",
    "relevance_grade",
    "query_source_title",
    "candidate_title",
    "query_attributes",
    "candidate_attributes",
    "query_image_path",
    "candidate_image_path",
    "primary_cause",
    "secondary_causes",
    "rule_fixability",
    "evidence",
]


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rankings = load_csv(resolve(config["source_rankings"]))
    judgments = load_csv(resolve(config["source_judgments"]))
    ranking_by_key = {
        (row["query_mode"], row["query_id"], row["product_id"]): row for row in rankings
    }
    judgment_by_key = {(row["query_id"], row["product_id"]): row for row in judgments}
    query_judgment = {}
    for row in judgments:
        query_judgment.setdefault(row["query_id"], row)
    catalog_by_id = {
        str(record["product_id"]): record
        for record in load_jsonl(resolve(config["source_catalog"]))
    }
    rows = []
    for case in config["cases"]:
        key = (case["query_mode"], case["query_id"], case["candidate_product_id"])
        ranking = ranking_by_key.get(key)
        judgment = judgment_by_key.get((case["query_id"], case["candidate_product_id"]))
        query_source = query_judgment.get(case["query_id"])
        candidate_source = catalog_by_id.get(case["candidate_product_id"])
        if ranking is None or query_source is None or candidate_source is None:
            raise ValueError(f"Bad Case 不在冻结排名或标签中：{case['case_id']}")
        if ranking["relevance_grade"] != "0" or (
            judgment is not None and judgment["relevance_grade"] != "0"
        ):
            raise ValueError(f"Bad Case 必须是人工 grade=0：{case['case_id']}")
        if ranking["query_category_l1"] != "3C数码":
            raise ValueError(f"Bad Case 必须属于3C查询：{case['case_id']}")
        rows.append(
            {
                "case_id": case["case_id"],
                "query_mode": case["query_mode"],
                "query_id": case["query_id"],
                "query_product_id": ranking["query_product_id"],
                "category_l2": ranking["query_category_l2"],
                "query_text": query_source["query_text"],
                "candidate_product_id": case["candidate_product_id"],
                "original_rank": ranking["rank"],
                "similarity_score": ranking["similarity_score"],
                "relevance_grade": ranking["relevance_grade"],
                "query_source_title": query_source["query_source_title"],
                "candidate_title": candidate_source["title"],
                "query_attributes": query_source["query_attributes"],
                "candidate_attributes": json.dumps(
                    candidate_source.get("attributes", {}), ensure_ascii=False, separators=(",", ":")
                ),
                "query_image_path": query_source["query_image_path"],
                "candidate_image_path": candidate_source["image_path"],
                "primary_cause": case["primary_cause"],
                "secondary_causes": "|".join(case["secondary_causes"]),
                "rule_fixability": case["rule_fixability"],
                "evidence": case["evidence"],
            }
        )
    return rows


def validate_selection(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if len(rows) != 20 or len({row["case_id"] for row in rows}) != 20:
        raise ValueError("Bad Case 必须恰好包含20条唯一案例。")
    if Counter(row["category_l2"] for row in rows) != {
        "耳机": 5,
        "键盘": 5,
        "鼠标": 5,
        "移动电源": 5,
    }:
        raise ValueError("四个3C二级品类必须各5条。")
    if Counter(row["query_mode"] for row in rows) != {"text": 10, "image": 10}:
        raise ValueError("text 与 image Bad Case 必须各10条。")
    labels = set(config["root_cause_labels"])
    if any(row["primary_cause"] not in labels for row in rows):
        raise ValueError("存在未登记的 Bad Case 根因。")


def build_markdown(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    causes = Counter(row["primary_cause"] for row in rows)
    fixability = Counter(row["rule_fixability"] for row in rows)
    lines = [
        "# 3C 检索 Bad Case 分析（20 条）",
        "",
        f"- 版本：`{config['version']}`",
        f"- 选择口径：{config['selection_policy']}",
        "- 数据边界：允许用冻结 test50 做一次定性根因分析；本报告不用于选择候选池、规则或权重。",
        "- 公平性：Image→Text 的标准品类和属性只用于离线归因，不进入正式规则重排。",
        "",
        "## 根因统计",
        "",
        "| 根因 | 数量 | 比例 |",
        "| --- | ---: | ---: |",
    ]
    for cause in config["root_cause_labels"]:
        count = causes[cause]
        lines.append(f"| `{cause}` | {count} | {count / len(rows):.0%} |")
    lines.extend(
        [
            "",
            "## 规则可修复性",
            "",
            "| 判断 | 数量 |",
            "| --- | ---: |",
        ]
    )
    for label, count in sorted(fixability.items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## 逐例记录",
            "",
            "| ID | 模式 | 品类 | 查询 | 原排名 | 错误候选 | 主因 | 规则可修复性 |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['query_mode']} | {row['category_l2']} | "
            f"{row['query_text']} | {row['original_rank']} | {row['candidate_title']} | "
            f"`{row['primary_cause']}` | `{row['rule_fixability']}` |"
        )
    lines.extend(["", "## 逐例依据", ""])
    for row in rows:
        lines.append(f"- **{row['case_id']}：** {row['evidence']}")
    lines.extend(
        [
            "## 结论",
            "",
            "- 文本查询中的明确品类、连接方式、规格数字和功能冲突，可由更深候选池上的规则重排修正，前提是相关商品已进入候选池。",
            "- 外观同质化、图片侧细粒度属性缺失和跨模态对齐不足，无法由当前正式 Image→Text 规则修正。",
            "- 标题污染、属性错位或容量声明异常属于源数据质量问题，不应靠提高规则权重掩盖。",
            "- 跨品类误召回在文本查询可由解析出的品类降权；图片查询缺少独立品类识别器时不能使用隐藏 metadata 修正。",
            "",
        ]
    )
    return "\n".join(lines)


def write_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    output_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category_l2"]].append(row)
    for category, cases in by_category.items():
        sheet = Image.new("RGB", (560, 5 * 300), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(cases):
            top = index * 300
            for column, field in enumerate(("query_image_path", "candidate_image_path")):
                image = Image.open(resolve(row[field])).convert("RGB")
                image.thumbnail((250, 250))
                cell = Image.new("RGB", (250, 250), "white")
                cell.paste(image, ((250 - image.width) // 2, (250 - image.height) // 2))
                sheet.paste(ImageOps.expand(cell, border=1, fill="black"), (10 + column * 275, top + 35))
            draw.text((10, top + 8), f"{row['case_id']} {row['query_mode']} rank={row['original_rank']} query", fill="black")
            draw.text((285, top + 8), f"candidate {row['candidate_product_id']}", fill="black")
        sheet.save(output_dir / f"{category}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contact-sheet-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("source_rankings", "source_rankings_sha256"),
        ("source_judgments", "source_judgments_sha256"),
        ("source_queries", "source_queries_sha256"),
        ("source_catalog", "source_catalog_sha256"),
    ):
        if file_sha256(resolve(config[path_key])) != config[hash_key]:
            raise ValueError(f"{path_key} SHA256 与 Bad Case 配置不一致。")
    rows = build_rows(config)
    validate_selection(config, rows)
    atomic_write_csv(resolve(config["csv_output"]), rows)
    atomic_write_text(resolve(config["markdown_output"]), build_markdown(config, rows))
    if args.contact_sheet_dir:
        write_contact_sheets(rows, args.contact_sheet_dir.resolve())
    print(json.dumps({"status": "completed", "case_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
