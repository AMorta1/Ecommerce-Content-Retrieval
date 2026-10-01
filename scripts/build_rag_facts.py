"""从 validation 构建可追溯的轻量 RAG 商品事实单元。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.rag import (  # noqa: E402
    SUPPORTED_TASKS,
    build_fact_units,
    load_fact_policy,
    load_quality_audit,
    select_top_k_facts,
)


DEFAULT_INPUT = PROJECT_ROOT / "data/processed/week1_v3/multimodal/validation.jsonl"
DEFAULT_AUDIT = PROJECT_ROOT / "reports/data/week2_rag_fact_quality_audit.csv"
DEFAULT_POLICY = PROJECT_ROOT / "configs/rag_fact_policy_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/week2_rag_v1/validation_fact_units.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports/generation/rag/validation_fact_build_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scope", default="validation")
    parser.add_argument("--dataset-version", default="week1_v3")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    audit_path = args.audit.resolve()
    policy_path = args.policy.resolve()
    policy = load_fact_policy(policy_path)
    records = load_jsonl(input_path)
    audit = load_quality_audit(audit_path, args.scope)
    record_ids = {str(record["product_id"]) for record in records}
    if record_ids != set(audit):
        raise ValueError(
            "输入商品与审计记录集合不一致："
            f"缺少审计={len(record_ids - set(audit))}，多余审计={len(set(audit) - record_ids)}"
        )

    source_sha256 = file_sha256(input_path)
    relative_source = input_path.relative_to(PROJECT_ROOT).as_posix()
    expected_sources = {
        item["scope"]: item for item in policy["design_basis"]["source_datasets"]
    }
    expected = expected_sources.get(args.scope)
    if expected is None or expected["sha256"] != source_sha256 or expected["count"] != len(records):
        raise ValueError("输入数据的数量或 SHA256 与事实策略不一致。")

    all_facts: list[dict[str, Any]] = []
    product_summaries = []
    selection_counts = {task: Counter() for task in SUPPORTED_TASKS}
    for record in records:
        product_id = str(record["product_id"])
        facts = build_fact_units(
            record,
            audit[product_id],
            policy,
            dataset_version=args.dataset_version,
            source_path=relative_source,
            source_sha256=source_sha256,
        )
        all_facts.extend(facts)
        task_summary = {}
        for task in SUPPORTED_TASKS:
            selection = select_top_k_facts(facts, policy, task=task, top_k=args.top_k)
            identity_count = len(selection["identity_facts"])
            selected_count = len(selection["selected_facts"])
            selection_counts[task]["identity_facts"] += identity_count
            selection_counts[task]["selected_facts"] += selected_count
            task_summary[task] = {
                "identity_fact_count": identity_count,
                "selected_fact_count": selected_count,
            }
        product_summaries.append(
            {
                "product_id": product_id,
                "audit_status": audit[product_id]["status"],
                "fact_count": len(facts),
                "quality_status_counts": dict(
                    sorted(Counter(fact["quality_status"] for fact in facts).items())
                ),
                "task_selection": task_summary,
            }
        )

    write_jsonl(args.output.resolve(), all_facts)
    report = {
        "version": policy.get("fact_build_version", "validation_fact_build_v1"),
        "status": "passed",
        "scope": args.scope,
        "dataset_version": args.dataset_version,
        "source": {
            "path": relative_source,
            "sha256": source_sha256,
            "record_count": len(records),
        },
        "audit": {
            "path": audit_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(audit_path),
        },
        "policy": {
            "path": policy_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(policy_path),
            "version": policy["version"],
        },
        "top_k": args.top_k
        if args.top_k is not None
        else policy["initial_retrieval_proposal"]["top_k"],
        "top_k_status": policy["initial_retrieval_proposal"]["top_k_status"],
        "fact_count": len(all_facts),
        "fact_role_counts": dict(sorted(Counter(f["fact_role"] for f in all_facts).items())),
        "quality_status_counts": dict(
            sorted(Counter(f["quality_status"] for f in all_facts).items())
        ),
        "mandatory_status_counts": dict(
            sorted(Counter(f.get("mandatory_status", "not_core") for f in all_facts).items())
        ),
        "task_selection_totals": {
            task: dict(counts) for task, counts in selection_counts.items()
        },
        "model_execution": False,
        "test100_execution": False,
        "products": product_summaries,
    }
    write_json(args.report.resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "products"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
