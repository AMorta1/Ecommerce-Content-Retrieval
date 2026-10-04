"""Archive the user-confirmed fixed test100 review; never call a model or tune."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_rag_project_test import (
    OUTPUT, WORKBOOK, VARIANTS, METRICS, DIAGNOSTICS, PREFERENCES,
    paired_data, validate_review_rows, summarize, verify_protocol,
)
from src.generation.lora_training import ROOT, atomic_json, digest, now, read_json, resolve
from src.generation.lora_validation_review import official_metrics
from import_lora_rag_ai_review import ALLOWED, sheet_snapshot


def readable_report(out, human, automatic, rows, mapping, changes):
    totals = human["versions"]
    a, b = [totals[v] for v in VARIANTS]
    percent = lambda x: f"{x * 100:.2f}%"
    lines = ["# LoRA + RAG 固定 test100 最终人工评测归档", "", human["test_history_disclosure"], "",
             "本轮100对 / 200份文案全部经用户最终确认；先保存匿名偏好汇总，再解盲。AI辅助初审、Codex修正、用户最终复核，不是独立双人标注。", "",
             "## 正式人工指标", "", "| 指标 | 冻结 LoRA | LoRA + RAG |", "|---|---:|---:|"]
    for label, key, numerator, denominator in (
        ("核心属性命中率", "core_attribute_hit_rate", "matched_attribute_total", "core_attribute_total"),
        ("通顺率", "fluency_pass_rate", "fluency_pass_count", "samples"),
        ("事实错误样本率", "factual_error_sample_rate", "factual_error_sample_count", "samples"),
    ):
        values = []
        for v in VARIANTS:
            t = totals[v]
            # Metric helper uses the stable names recorded in the JSON; use rates for counts if absent.
            n = t.get(numerator, round(t[key] * (t["core_attribute_total"] if denominator == "core_attribute_total" else 100)))
            d = t["core_attribute_total"] if denominator == "core_attribute_total" else 100
            values.append(f"{n}/{d} = {percent(t[key])}")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += [f"| 事实错误总数 | {a['factual_error_total']} | {b['factual_error_total']} |",
              f"| 平均事实错误数/件 | {a['average_factual_errors_per_sample']:.2f} | {b['average_factual_errors_per_sample']:.2f} |",
              "", "## 匿名表达偏好（汇总后解盲）", "",
              "| 维度 | LoRA更好 | 组合更好 | 持平 |", "|---|---:|---:|---:|"]
    labels = ("标题表达质量", "卖点结构", "短详情自然度", "整体电商专业度")
    for p, label in zip(PREFERENCES, labels):
        votes = human["preferences"][p]
        lines.append(f"| {label} | {votes[VARIANTS[0]]} | {votes[VARIANTS[1]]} | {votes['tie']} |")
    lines += ["", "## 分品类正式指标", "",
              "| 二级品类 | 件数 | LoRA / 组合核心命中 | LoRA / 组合通顺率 | LoRA / 组合错误样本率 |",
              "|---|---:|---|---|---|"]
    for category in human["by_category"][VARIANTS[0]]:
        first, second = [human["by_category"][v][category] for v in VARIANTS]
        count = sum(r["category_l2"] == category and r["variant"] == VARIANTS[0] for r in rows)
        lines.append(f"| {category} | {count} | {first['matched_attribute_total']}/{first['core_attribute_total']} / {second['matched_attribute_total']}/{second['core_attribute_total']} | {percent(first['fluency_pass_rate'])} / {percent(second['fluency_pass_rate'])} | {percent(first['factual_error_sample_rate'])} / {percent(second['factual_error_sample_rate'])} |")
    lines += ["", "## 分品类匿名偏好（解盲后）", "",
              "每格依次为 LoRA更好 / 组合更好 / 持平；少量品类样本不作总体外推。", "",
              "| 品类 | 标题 | 卖点结构 | 短详情自然度 | 整体专业度 |", "|---|---|---|---|---|"]
    ws = load_workbook(out / WORKBOOK)["盲评成对复核"]
    cols = {c.value: c.column for c in ws[1]}
    category_votes = {}
    for i, pair in enumerate(mapping, 3):
        category = ws.cell(i, cols["category_l2"]).value
        counts = category_votes.setdefault(category, {p: Counter() for p in PREFERENCES})
        for p in PREFERENCES:
            vote = ws.cell(i, cols[p]).value
            counts[p]["tie" if vote == "持平" else pair[vote[0]]] += 1
    for category in sorted(category_votes):
        cells = [" / ".join(str(category_votes[category][p][v]) for v in (*VARIANTS, "tie")) for p in PREFERENCES]
        lines.append("| " + category + " | " + " | ".join(cells) + " |")
    lines += ["", "## 自动工程与辅助指标", "", "| 指标 | 冻结 LoRA | LoRA + RAG |", "|---|---:|---:|"]
    for task in ("title", "selling_points", "short_description", "three_task_total"):
        values = [automatic["versions"][v]["latency_seconds"][task]["mean"] for v in VARIANTS]
        lines.append(f"| {task} 平均耗时（秒） | {values[0]:.3f} | {values[1]:.3f} |")
    for label, key in (("组装结构成功/100", "assembled_structure_success"), ("空任务/300", "empty_tasks"),
                       ("max-token触顶任务/300", "max_token_tasks"), ("疑似评价样本（非事实错误率）", "claim_suspect_samples_NOT_FACTUAL_RATE"),
                       ("疑似数值样本（非事实错误率）", "number_suspect_samples_NOT_FACTUAL_RATE")):
        values = [automatic["versions"][v][key] for v in VARIANTS]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    for v in VARIANTS:
        s = automatic["versions"][v]
        lines += ["", f"{v}：逐任务及总耗时 {json.dumps(s['latency_seconds'], ensure_ascii=False)}；逐任务解析 {s['task_parse_success']}；",
                  f"字面核心覆盖（非正式人工指标）{s['literal_core_coverage_NOT_HUMAN']}；Mandatory tokenizer后注入 {s['mandatory_core_injection_per_task']}；峰值allocated {s['peak_allocated_mib']:.1f}MiB。"]
    lines += ["", "历史control与新组合耗时非同期测量，不作纯RAG因果归因。", "",
              "## 人工诊断（不替代正式三指标）", "", "| 诊断阳性文案数 | 冻结 LoRA | LoRA + RAG |", "|---|---:|---:|"]
    for d in DIAGNOSTICS:
        values = [sum(str(r[d]) == "1" for r in rows if r["variant"] == v) for v in VARIANTS]
        lines.append(f"| {d} | {values[0]} | {values[1]} |")
    lines += ["", "## 重点问题与边界案例", ""]
    for case in read_json(out / "human_diagnostic_cases.json")["cases"]:
        lines += [f"- {case['pair_id']} / {case['product_id']} / {case['category_l2']} / {case['variant']}：核心命中{case['matched_attribute_count']}/{case['core_attribute_count']}、通顺{case['fluency_pass']}、事实错误{case['factual_error_count']}。{case['review_notes']}"]
    lines += ["", "源数据争议与模型新增错误分开：CT070连接方式源歧义不新增事实错误；CT094源身份CONFLICT导致输入撤回，但完整源仍可支持多U口的一般多设备充电表述。字段提示泄漏及空身份字典记录为结构/通顺失败，不自动折算事实错误。CT053家庭清洁边界按用户最终确认的严格评分保留。", "",
              "## 结论与限制", "",
              f"组合相对冻结LoRA：核心覆盖差值 {(b['core_attribute_hit_rate'] - a['core_attribute_hit_rate']) * 100:+.2f} 个百分点；通顺率差值 {(b['fluency_pass_rate'] - a['fluency_pass_rate']) * 100:+.2f} 个百分点；事实错误样本率差值 {(b['factual_error_sample_rate'] - a['factual_error_sample_rate']) * 100:+.2f} 个百分点。",
              ("本轮没有观察到组合的核心属性覆盖收益，不能将Mandatory完整注入等同于输出完整覆盖。" if b['core_attribute_hit_rate'] <= a['core_attribute_hit_rate'] else "本轮观察到组合核心属性覆盖提高；它是本批固定样本上的观察，不作为新的未观察测试泛化证据。"),
              "覆盖、事实性与表达偏好分别解释；多数持平或少量胜负不能宣称稳定总体表达收益。偏好理由涉及参数选择/冗余减少时，不等同纯语言流畅性改善。",
              "本轮control是同一冻结LoRA输出在新的匿名配对工作簿中重新复核，其标签不直接继承旧LoRA-only人工结果；既有正式结果完全不修改，若标签不同须保留评审语境与判断依据，不择优替换旧指标。",
              "这100件此前已被观察，本轮只是固定方案追加正式对照，不是新的未观察盲测。版本名匿名不等于严格双盲，实际输入与区块泄漏可能透露条件。四区上下文和必要system接口适配是联合处理，不能仅归因为检索算法。",
              f"用户相对导入草稿的评分/偏好/备注修改数：{len(changes)}。所有100行均已确认；reviewer空值仅补来源元数据，未更改用户指标。",
              "本阶段结束；无新增模型调用、训练、Prompt/policy/数据/解码修改或重跑；不进入原RAG v2 P3，不读取32条holdout归档，不执行其生成，不commit/push。结果无论好坏只归档。", "",
              "## 最终归档", "",
              "final_test_report.md：冻结入口的正式总报告；human_metrics_unblinded.json：正式三指标/偏好/分品类；blind_summary.json：先匿名汇总；human_review_final.csv：200份最终文案评分；human_diagnostic_cases.json：诊断；human_review_confirmation_record.json：用户确认和元数据补全证据；final_review_manifest.json及final_review_checksums.sha256：最终交付hash链（含既有final_test_manifest，不改写旧manifest）。"]
    (out / "final_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out = resolve(OUTPUT)
    path = out / WORKBOOK
    backup = path.with_name(path.stem + ".user_completed.xlsx")
    temporary = path.with_name(path.stem + ".confirmation.tmp.xlsx")
    for name in ("blind_summary.json", "human_metrics_unblinded.json", "human_review_final.csv", "final_test_manifest.json",
                 "human_review_confirmation_record.json", "final_review_manifest.json", "final_review_report.md"):
        if (out / name).exists():
            raise FileExistsError("Do not overwrite an existing finalization")
    if backup.exists() or temporary.exists():
        raise FileExistsError("Existing finalization backup/temporary: inspect before continuing")
    _, _, samples, versions, mapping = paired_data()
    for entry in read_json(out / "review_manifest.json")["files"]:
        if digest(resolve(entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen review generation changed")
    for entry in read_json(out / "automatic_phase_manifest.json")["files"]:
        original = out / "lora_rag_project_test100_blind_review.pre_ai_import.xlsx" if resolve(entry["path"]) == path else resolve(entry["path"])
        if digest(original) != entry["sha256"]:
            raise ValueError("Historical automatic archive changed")
    imported = read_json(out / "ai_review_import_manifest_v1.json")
    for p, key in ((out / "ai_review_amendments_v1.json", "amendments_sha256"),
                   (resolve(imported["source_archive"]), "source_archive_sha256"),
                   (resolve(imported["original_backup"]), "backup_sha256"),
                   (resolve("scripts/import_lora_rag_test100_ai_review.py"), "importer_sha256"),
                   (resolve("scripts/import_lora_rag_ai_review.py"), "helper_sha256")):
        if digest(p) != imported[key]:
            raise ValueError("AI-import provenance changed")
    before_sha = digest(path)
    book = load_workbook(path, data_only=False)
    ws = book["盲评成对复核"]
    original_book = load_workbook(resolve(imported["original_backup"]))
    if book.sheetnames != original_book.sheetnames:
        raise ValueError("Original sheets changed")
    original_ws = original_book[ws.title]
    columns = {c.value: c.column for c in ws[1]}
    for sheet in original_book:
        if sheet.title != ws.title and sheet_snapshot(book[sheet.title]) != sheet_snapshot(sheet):
            raise ValueError("Guide/hidden mapping/automatic sheets changed")
    for r in range(1, 103):
        for key, c in columns.items():
            if (r <= 2 or key not in ALLOWED) and ws.cell(r, c).value != original_ws.cell(r, c).value:
                raise ValueError("Frozen review source/output/header changed")
    if ws.max_row != 102 or any(str(ws.cell(r, columns["review_confirmed"]).value).strip() != "1" for r in range(3, 103)):
        raise ValueError("All 100 rows must be explicitly user-confirmed")
    raw_values = list(ws.values)
    sheets = {s.title: sheet_snapshot(s) for s in book if s.title != ws.title}
    supplied_reviewers = Counter(ws.cell(r, columns["reviewer"]).value for r in range(3, 103))
    filled = []
    for r in range(3, 103):
        cell = ws.cell(r, columns["reviewer"])
        if not str(cell.value or "").strip():
            cell.value = "用户复核确认（ChatGPT初审 / Codex修正）"
            cell.data_type = "s"
            filled.append(ws.cell(r, columns["pair_id"]).value)
    validate_review_rows(ws, samples, versions, mapping)
    incoming = load_workbook(resolve(imported["source_archive"]))[ws.title]
    amendments = read_json(out / "ai_review_amendments_v1.json")
    changes = []
    for r in range(3, 103):
        pair = ws.cell(r, columns["pair_id"]).value
        patches = amendments["corrections"].get(pair, {}).get("fields", {})
        for key in sorted(ALLOWED - {"reviewer", "review_confirmed"}):
            expected = patches[key]["value"] if key in patches else incoming.cell(r, columns[key]).value
            actual = ws.cell(r, columns[key]).value
            if actual != expected:
                changes.append({"pair_id": pair, "field": key, "ai_import_value": expected, "user_final_value": actual})
    if digest(path) != before_sha:
        raise ValueError("Concurrent workbook update")
    shutil.copy2(path, backup)
    if digest(backup) != before_sha:
        raise ValueError("User backup hash mismatch")
    book.save(temporary)
    reopened = load_workbook(temporary)
    if reopened.sheetnames != book.sheetnames or any(sheet_snapshot(reopened[t]) != s for t, s in sheets.items()):
        raise ValueError("Non-review sheets changed on save")
    saved = reopened[ws.title]
    for r, values in enumerate(raw_values, 1):
        for c, value in enumerate(values, 1):
            if r >= 3 and c == columns["reviewer"]:
                continue
            if saved.cell(r, c).value != value:
                raise ValueError("Metadata completion changed scores/notes/fixed data")
    validate_review_rows(saved, samples, versions, mapping)
    if digest(path) != before_sha:
        raise ValueError("Concurrent workbook update before replacement")
    os.replace(temporary, path)
    atomic_json(out / "human_review_confirmation_record.json", {
        "status": "user_confirmed_100_rows", "created_at_utc": now(), "user_message": "我已复核完成",
        "confirmed_rows": 100, "reviewed_documents": 200, "reviewer_metadata_completed_pairs": filled,
        "supplied_reviewer_values": [{"value": k, "count": v} for k, v in supplied_reviewers.items()],
        "metadata_completion_changes_scores_preferences_or_notes": False,
        "user_saved_workbook": backup.relative_to(ROOT).as_posix(), "user_saved_sha256": before_sha,
        "completed_workbook_sha256": digest(path), "ai_import_manifest_sha256": digest(out / "ai_review_import_manifest_v1.json"),
        "user_edits_vs_imported_ai_draft": changes, "finalization_script_sha256": digest(Path(__file__)),
        "provenance": "AI-assisted initial judgments, Codex amendment, user final review/confirmation; not independent double-human review"})
    # Frozen summarizer writes the anonymous aggregate BEFORE applying version identities.
    summarize()
    human = read_json(out / "human_metrics_unblinded.json")
    blind = read_json(out / "blind_summary.json")
    with (out / "human_review_final.csv").open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    if len(exported) != 200 or len({(r["variant"], r["product_id"]) for r in exported}) != 200:
        raise ValueError("Final CSV scope mismatch")
    for v in VARIANTS:
        if official_metrics([r for r in exported if r["variant"] == v]) != human["versions"][v]:
            raise ValueError("Exported scores do not reproduce final metrics")
    if blind["workbook_sha256"] != digest(path) or human["blind_summary_sha256"] != digest(out / "blind_summary.json"):
        raise ValueError("Anonymous-before-unblind hash chain mismatch")
    for p in PREFERENCES:
        if sum(human["preferences"][p].values()) != 100 or sum(blind["preferences"][p].values()) != 100:
            raise ValueError("Preference total mismatch")
    readable_report(out, human, read_json(out / "automatic_summary.json"), exported, mapping, changes)
    _, _, _, frozen = verify_protocol()
    paths = {resolve(item["path"]) for item in frozen["files"]}
    paths.update(p for p in out.rglob("*") if p.is_file())
    paths.update((Path(__file__).resolve(), resolve("scripts/import_lora_rag_test100_ai_review.py"), resolve("scripts/import_lora_rag_ai_review.py")))
    files = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sorted(paths)]
    final = out / "final_review_manifest.json"
    atomic_json(final, {"status": "completed_user_confirmed_additional_fixed_test100", "created_at_utc": now(),
        "products": 100, "reviewed_documents": 200, "parameter_updates": 0, "new_inference_calls_in_finalization": 0,
        "previously_observed_test": True, "fresh_unobserved_blind_test": False,
        "holdout_archive_read": False, "holdout_generation": False, "rag_v2_historical_holdout_status": "frozen_not_executed",
        "original_rag_v2_p3_entered": False, "historical_formal_labels_modified": False,
        "human_provenance": "AI-assisted draft, Codex amendment, user final review/confirmation",
        "protocol_manifest_sha256": digest(out / "protocol_manifest.json"), "workbook_sha256": digest(path),
        "anonymous_summary_sha256": digest(out / "blind_summary.json"), "human_metrics_sha256": digest(out / "human_metrics_unblinded.json"),
        "final_report_sha256": digest(out / "final_review_report.md"), "user_edits_vs_imported_ai_draft": changes, "files": files})
    entries = files + [{"path": final.relative_to(ROOT).as_posix(), "sha256": digest(final)}]
    (out / "final_review_checksums.sha256").write_text("".join(f"{e['sha256']}  {e['path']}\n" for e in entries), encoding="utf-8")
    print(json.dumps({"status": "completed_stop_no_retuning", "reviewer_metadata_completed": len(filled),
                      "user_label_edits": len(changes), "versions": human["versions"], "preferences": human["preferences"],
                      "archive_files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
