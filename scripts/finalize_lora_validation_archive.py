"""Close P7 automatic-only archive after report, without human/test inference."""
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_validation import verify_protocol
from src.generation.lora_training import atomic_json, digest, now, read_json, resolve
from scripts.review_lora_project_validation_quick32 import verify_sample


def main():
    cfg, protocol = verify_protocol()
    _, _, quick_out, _ = verify_sample()
    out = resolve(cfg["output"])
    if (out / "delivery_manifest.json").exists():
        raise FileExistsError("No delivery archive overwrite")
    automatic = read_json(out / "automatic_phase_manifest.json")
    for item in automatic["files"]:
        if digest(resolve(item["path"])) != item["sha256"]:
            raise ValueError(f"Archived output changed: {item['path']}")
    review = read_json(quick_out / "review_manifest.json")
    if digest(quick_out / "project_validation32_blind_review.xlsx") != review["initial_workbook_sha256"]:
        raise ValueError("Initial human workbook changed before delivery")
    if any((quick_out / name).exists() for name in ("human_metrics_unblinded.json", "blind_summary.json", "human_review_final.csv")):
        raise ValueError("Delivery is automatic-phase only")
    paths = {resolve(i["path"]) for i in protocol["files"] + automatic["files"]}
    paths.update(p for p in out.rglob("*") if p.is_file())
    paths.update((resolve("scripts/finalize_lora_validation_archive.py"), resolve("tests/test_lora_validation_quick32.py")))
    files = [{"path": str(p.relative_to(resolve("."))).replace("\\", "/"), "sha256": digest(p)} for p in sorted(paths)]
    manifest = {"status": "generation_complete_quick32_human_review_pending", "created_at_utc": now(),
        "generation_products_per_model": 200, "tasks_per_model": 600, "human_sample_products": 32,
        "tests": {"passed": 59, "failed": 0, "scope": "focused_P5_P6_P7_unit_tests_synthetic_review_fixtures"},
        "artifact_writer_environment": {"python": sys.version, "openpyxl": importlib.metadata.version("openpyxl")},
        "inference_protocol_sha256": digest(out / "protocol_manifest.json"),
        "sampling_manifest_sha256": digest(quick_out / "sampling_manifest.json"),
        "automatic_phase_manifest_sha256": digest(out / "automatic_phase_manifest.json"),
        "human_formal_metrics_exist": False, "automatic_test_promotion": False,
        "project_test_read": False, "rag_holdout_files_read": False, "rag_v2_generation": False, "parameter_updates": 0,
        "files": files, "workbook_hash_role": "initial_blank_workbook_expected_to_change_only_by_human_fill"}
    path = out / "delivery_manifest.json"
    atomic_json(path, manifest)
    rows = files + [{"path": str(path.relative_to(resolve("."))).replace("\\", "/"), "sha256": digest(path)}]
    (out / "delivery_checksums.sha256").write_text("".join(f"{i['sha256']}  {i['path']}\n" for i in rows), encoding="utf-8")
    print(f"Delivery frozen: {len(files)} files; human review pending; no test.")


if __name__ == "__main__":
    main()
