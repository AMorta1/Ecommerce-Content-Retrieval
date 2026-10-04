"""Run the frozen rerank_v1 formal test50 experiment exactly once.

The command requires either ``--preflight-only`` or the explicit
``--confirm-formal-test50`` execution flag. Preflight never loads ERNIE-ViL and
never reads model outputs from test50.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_catalog, load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.evaluation import (  # noqa: E402
    annotation_progress,
    count_assistant_draft_rows,
    load_annotation_rows,
)
from src.retrieval.faiss_index import load_index, search  # noqa: E402
from src.retrieval.formal_evaluation import (  # noqa: E402
    evaluate_complete_rankings,
    parse_complete_relevance,
    validate_query_review,
)
from src.retrieval.rerank import rerank_candidates  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_rerank_formal_v1.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "reports" / "retrieval" / "rerank" / "rerank_v1_formal_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--confirm-formal-test50", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(configured_path: str | Path) -> Path:
    return resolve_project_path(PROJECT_ROOT, configured_path)


def assert_nonnegative_similarities(
    scores: Iterable[float], *, query_id: str
) -> tuple[float, float]:
    """Fail closed before a multiplicative category penalty can be inverted."""
    values = [float(score) for score in scores]
    if not values:
        raise ValueError(f"Query {query_id} has an empty candidate pool.")
    minimum = min(values)
    maximum = max(values)
    if minimum < 0:
        raise ValueError(
            f"Query {query_id} contains negative similarity {minimum:.9f}; "
            "multiplicative category factors would invert the penalty direction. "
            "Formal execution aborted without changing the frozen formula."
        )
    return minimum, maximum


def build_gallery_candidate_pool(
    scores: Sequence[float],
    indexes: Sequence[int],
    metadata: list[dict[str, Any]],
    catalog_by_id: dict[str, dict[str, Any]],
    query: dict[str, Any],
    *,
    gallery_split: str,
    candidate_pool_depth: int,
    score_precision: int,
) -> list[dict[str, Any]]:
    """Filter full-index results to the frozen gallery and return Top-N candidates."""
    candidates: list[dict[str, Any]] = []
    raw_scores: list[float] = []
    for source_rank, (score, metadata_index) in enumerate(zip(scores, indexes), start=1):
        product = metadata[int(metadata_index)]
        if product["split"] != gallery_split or product["product_id"] == query["product_id"]:
            continue
        catalog_record = catalog_by_id[product["product_id"]]
        raw_scores.append(float(score))
        candidates.append(
            {
                "candidate_rank": len(candidates) + 1,
                "source_index_rank": source_rank,
                "similarity_score": round(float(score), score_precision),
                "product_id": product["product_id"],
                "title": product["title"],
                "candidate_category_l1": product["category_l1"],
                "candidate_category_l2": product["category_l2"],
                "candidate_attributes": catalog_record.get("attributes", {}),
            }
        )
        if len(candidates) == candidate_pool_depth:
            break
    if len(candidates) != candidate_pool_depth:
        raise ValueError(
            f"Query {query['query_id']} has only {len(candidates)} eligible gallery candidates; "
            f"expected {candidate_pool_depth}."
        )
    assert_nonnegative_similarities(raw_scores, query_id=query["query_id"])
    return candidates


def load_baseline_image_passthrough(
    path: Path, *, query_ids: set[str], cutoff: int
) -> dict[str, list[dict[str, str]]]:
    """Read the frozen Image-to-Text Top-10 without using hidden query metadata to score."""
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["query_mode"] == "image"]
    grouped: dict[str, list[dict[str, str]]] = {query_id: [] for query_id in query_ids}
    for row in rows:
        query_id = row["query_id"]
        if query_id in grouped:
            grouped[query_id].append(row)
    for query_id, query_rows in grouped.items():
        query_rows.sort(key=lambda row: int(row["rank"]))
        expected_ranks = list(range(1, cutoff + 1))
        if [int(row["rank"]) for row in query_rows] != expected_ranks:
            raise ValueError(f"Frozen Image-to-Text ranking is incomplete for {query_id}.")
    return grouped


def evaluate_breakdowns(
    rankings: dict[str, list[str]],
    relevance: dict[str, dict[str, int]],
    queries: list[dict[str, Any]],
    cutoff: int,
    threshold: int,
) -> dict[str, Any]:
    query_by_id = {query["query_id"]: query for query in queries}
    result = {
        "overall": evaluate_complete_rankings(rankings, relevance, cutoff, threshold),
        "by_category_l1": {},
        "by_category_l2": {},
    }
    for field in ("category_l1", "category_l2"):
        for value in sorted({query[field] for query in queries}):
            query_ids = {
                query_id for query_id, query in query_by_id.items() if query[field] == value
            }
            result[f"by_{field}"][value] = evaluate_complete_rankings(
                {query_id: rankings[query_id] for query_id in query_ids},
                {query_id: relevance[query_id] for query_id in query_ids},
                cutoff,
                threshold,
            )["macro_average"]
    return result


def verify_path_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 changed: expected {expected}, got {actual}.")


def verify_model_cache(manifest: dict[str, Any]) -> Path:
    model = manifest["model_identity"]
    env_name = model["cache_root_environment_variable"]
    configured_root = os.environ.get(env_name)
    if not configured_root:
        raise EnvironmentError(
            f"{env_name} must point to the prepared PaddleNLP cache before formal execution."
        )
    cache_root = Path(configured_root).resolve()
    for item in model["local_artifacts"]:
        verify_path_hash(cache_root / item["relative_path"], item["sha256"], item["relative_path"])
    return cache_root


def preflight(
    config_path: Path,
    manifest_path: Path,
    *,
    require_model_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    config = load_json(config_path)
    manifest = load_json(manifest_path)

    if config["version"] != manifest["version"]:
        raise ValueError("Formal config and manifest versions differ.")
    if config["formal_test50_executed"] or manifest["formal_test50_executed"]:
        raise ValueError("The frozen manifest says formal test50 was already executed.")
    if config["status"] != "frozen_pending_single_test50_execution":
        raise ValueError("Formal config is not in the frozen pending state.")
    verify_path_hash(config_path, manifest["formal_config"]["sha256"], "formal config")

    selected = config["selected_parameters"]
    if int(selected["candidate_pool_depth"]) <= int(selected["metric_cutoff"]):
        raise ValueError("Candidate pool must be larger than the Top-10 metric cutoff.")
    if config["query_mode_policy"] != {
        "text": "rule_rerank_from_query_text",
        "image": "baseline_passthrough_no_image_query_signals",
    }:
        raise ValueError("Formal query-mode policy changed.")
    if config["negative_similarity_safety"]["action"] != "abort_formal_execution":
        raise ValueError("Negative-similarity safety must fail closed.")

    for item in manifest["frozen_project_inputs"]:
        path = project_path(item["path"])
        verify_path_hash(path, item["sha256"], item["path"])
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"{item['path']} size changed.")
    for item in manifest["runtime_code"]:
        verify_path_hash(project_path(item["path"]), item["sha256"], item["path"])

    policy_path = project_path(config["rerank_policy_source"])
    verify_path_hash(policy_path, config["rerank_policy_source_sha256"], "rerank policy source")
    policy = load_json(policy_path)
    if policy["category_weights"] != selected["category_weights"]:
        raise ValueError("Selected category factors differ from the frozen policy source.")
    validation_report = load_json(project_path(config["validation_evidence"]["result_path"]))
    validation_selected = validation_report["selected"]
    for key in ("candidate_pool_depth", "attribute_weight", "conflict_weight"):
        if validation_selected[key] != selected[key]:
            raise ValueError(f"Formal {key} differs from the selected validation result.")
    if validation_report["test50_executed"]:
        raise ValueError("Validation report unexpectedly says test50 was executed.")

    baseline_manifest = load_json(project_path(manifest["baseline_reference"]["archive_manifest"]))
    baseline_metric_classes = baseline_manifest["retrieval_baseline"][
        "metric_source_classification"
    ]
    if (
        baseline_metric_classes["prd_required"]
        != config["metric_classification"]["prd_primary_frozen_metrics"]
        or baseline_metric_classes["repository_supplemental"]
        != config["metric_classification"]["repository_supplemental_frozen_metrics"]
    ):
        raise ValueError("Formal metric classification differs from the frozen Baseline manifest.")

    for output_path in config["outputs"].values():
        if project_path(output_path).exists():
            raise FileExistsError(f"Formal output already exists; overwrite refused: {output_path}")
    if require_model_cache:
        verify_model_cache(manifest)
    return config, manifest, policy


def git_runtime_identity() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "git_commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "worktree_dirty": bool(command("status", "--porcelain")),
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Formal ranking output cannot be empty.")
    with path.open("x", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_formal_test50(
    config: dict[str, Any], manifest: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    evaluation_config = load_json(project_path(config["baseline_evaluation_config"]))
    retrieval_config = load_json(project_path(config["retrieval_config"]))
    if (
        evaluation_config["queries_path"] != config["queries_path"]
        or evaluation_config["annotation_path"] != config["relevance_judgments_path"]
        or int(evaluation_config["metric_cutoff"])
        != int(config["selected_parameters"]["metric_cutoff"])
    ):
        raise ValueError("Formal config no longer matches the frozen Baseline evaluation scope.")
    queries = load_jsonl(project_path(config["queries_path"]))
    query_ids = {query["query_id"] for query in queries}
    if len(queries) != int(config["query_count"]):
        raise ValueError("Frozen query count changed.")

    with project_path(config["query_review_path"]).open(
        encoding="utf-8-sig", newline=""
    ) as input_file:
        validate_query_review(list(csv.DictReader(input_file)), queries)
    annotation_rows = load_annotation_rows(project_path(config["relevance_judgments_path"]))
    progress = annotation_progress(annotation_rows)
    if progress["remaining_rows"] or count_assistant_draft_rows(annotation_rows):
        raise ValueError("Frozen relevance judgments are not fully human reviewed.")
    threshold = int(config["positive_grade_threshold"])
    relevance = parse_complete_relevance(annotation_rows, threshold)
    if set(relevance) != query_ids:
        raise ValueError("Frozen queries and relevance judgments differ.")

    catalog = load_catalog(project_path(retrieval_config["dataset_path"]), PROJECT_ROOT)
    catalog_by_id = {row["product_id"]: row for row in catalog}
    metadata = load_metadata(project_path(retrieval_config["metadata_path"]))
    image_index = load_index(project_path(retrieval_config["index_path"]))
    if len(catalog) != len(metadata) or len(metadata) != int(image_index.ntotal):
        raise ValueError("Catalog, metadata, and image index sizes differ.")

    # Deferred import ensures preflight cannot initialize Paddle or the model.
    from src.retrieval.ernie_vil import ErnieVilEmbedder

    started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=retrieval_config["model_name"],
        device=retrieval_config["device"],
        batch_size=int(retrieval_config["batch_size"]),
    )
    text_features = embedder.encode_texts([query["query_text"] for query in queries])
    all_scores, all_indexes = search(image_index, text_features, top_k=int(image_index.ntotal))

    selected = config["selected_parameters"]
    pool_depth = int(selected["candidate_pool_depth"])
    cutoff = int(selected["metric_cutoff"])
    text_rankings: dict[str, list[str]] = {}
    ranking_rows: list[dict[str, Any]] = []
    formal_similarity_minimum = float("inf")
    formal_similarity_maximum = float("-inf")
    for query_index, query in enumerate(queries):
        candidates = build_gallery_candidate_pool(
            all_scores[query_index],
            all_indexes[query_index],
            metadata,
            catalog_by_id,
            query,
            gallery_split=config["gallery_split"],
            candidate_pool_depth=pool_depth,
            score_precision=int(selected["similarity_score_decimal_places"]),
        )
        formal_similarity_minimum = min(
            formal_similarity_minimum,
            min(float(row["similarity_score"]) for row in candidates),
        )
        formal_similarity_maximum = max(
            formal_similarity_maximum,
            max(float(row["similarity_score"]) for row in candidates),
        )
        reranked = rerank_candidates(
            "text",
            candidates,
            policy,
            query_text=query["query_text"],
            candidate_pool_depth=pool_depth,
            attribute_weight=float(selected["attribute_weight"]),
            conflict_weight=float(selected["conflict_weight"]),
        )[:cutoff]
        text_rankings[query["query_id"]] = [row["product_id"] for row in reranked]
        for rank, row in enumerate(reranked, start=1):
            ranking_rows.append(
                {
                    "query_mode": "text",
                    "query_id": query["query_id"],
                    "query_product_id": query["product_id"],
                    "rank": rank,
                    "original_candidate_rank": row["original_rank"],
                    "source_index_rank": row["source_index_rank"],
                    "similarity_score": row["similarity_score"],
                    "category_factor": row["category_factor"],
                    "category_reason": row["category_reason"],
                    "attribute_match_ratio": row["attribute_match_ratio"],
                    "attribute_conflict_ratio": row["attribute_conflict_ratio"],
                    "matched_signals": json.dumps(row["matched_signals"], ensure_ascii=False),
                    "conflicted_signals": json.dumps(row["conflicted_signals"], ensure_ascii=False),
                    "rerank_score": round(float(row["rerank_score"]), 9),
                    "product_id": row["product_id"],
                    "title": row["title"],
                    "candidate_category_l1": row["candidate_category_l1"],
                    "candidate_category_l2": row["candidate_category_l2"],
                    "relevance_grade": relevance[query["query_id"]].get(row["product_id"], 0),
                    "rerank_policy": "rule_rerank_from_query_text",
                }
            )

    image_rows_by_query = load_baseline_image_passthrough(
        project_path(config["baseline_rankings_path"]), query_ids=query_ids, cutoff=cutoff
    )
    image_rankings: dict[str, list[str]] = {}
    for query in queries:
        query_rows = image_rows_by_query[query["query_id"]]
        image_rankings[query["query_id"]] = [row["product_id"] for row in query_rows]
        for row in query_rows:
            ranking_rows.append(
                {
                    "query_mode": "image",
                    "query_id": row["query_id"],
                    "query_product_id": row["query_product_id"],
                    "rank": int(row["rank"]),
                    "original_candidate_rank": int(row["rank"]),
                    "source_index_rank": int(row["source_index_rank"]),
                    "similarity_score": row["similarity_score"],
                    "category_factor": "",
                    "category_reason": "",
                    "attribute_match_ratio": "",
                    "attribute_conflict_ratio": "",
                    "matched_signals": "[]",
                    "conflicted_signals": "[]",
                    "rerank_score": "",
                    "product_id": row["product_id"],
                    "title": row["title"],
                    "candidate_category_l1": row["candidate_category_l1"],
                    "candidate_category_l2": row["candidate_category_l2"],
                    "relevance_grade": row["relevance_grade"],
                    "rerank_policy": "baseline_passthrough_no_image_query_signals",
                }
            )

    metrics_by_mode = {
        "text": evaluate_breakdowns(text_rankings, relevance, queries, cutoff, threshold),
        "image": evaluate_breakdowns(image_rankings, relevance, queries, cutoff, threshold),
    }
    baseline_metrics = load_json(project_path(config["baseline_metrics_path"]))
    report = {
        "status": "formal_test50_single_execution_completed",
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formal_test50_executed": True,
        "runtime_code_identity": git_runtime_identity(),
        "query_count": len(queries),
        "gallery_split": config["gallery_split"],
        "query_mode_policy": config["query_mode_policy"],
        "selected_parameters": selected,
        "metric_classification": config["metric_classification"],
        "metrics_by_query_mode": metrics_by_mode,
        "baseline_metrics_by_query_mode": baseline_metrics["metrics_by_query_mode"],
        "similarity_safety": {
            "text_candidate_pool_minimum": formal_similarity_minimum,
            "text_candidate_pool_maximum": formal_similarity_maximum,
            "negative_similarity_count": 0,
            "action_if_negative": "abort_formal_execution",
        },
        "annotation_progress": progress,
        "annotation_grade_counts": dict(
            sorted(Counter(int(row["relevance_grade"]) for row in annotation_rows).items())
        ),
        "timing_seconds": round(time.perf_counter() - started, 3),
        "manifest_sha256": file_sha256(project_path(config["manifest_path"])),
        "frozen_config_sha256": manifest["formal_config"]["sha256"],
        "limitations": config["reporting_constraints"],
    }
    write_csv_exclusive(project_path(config["outputs"]["rankings"]), ranking_rows)
    write_json_exclusive(project_path(config["outputs"]["metrics"]), report)
    return report


def main() -> None:
    args = parse_args()
    config, manifest, policy = preflight(
        args.config,
        args.manifest,
        require_model_cache=args.confirm_formal_test50,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "version": config["version"],
                    "formal_test50_executed": False,
                    "model_loaded": False,
                    "test50_rankings_generated": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(run_formal_test50(config, manifest, policy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
