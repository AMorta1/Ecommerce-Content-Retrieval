"""Build a small, traceable experiment set using a disk-backed candidate table."""

from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlsplit, urlunsplit

from .pv_parser import normalize_text, parse_pv
from .streaming import iter_products


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def canonical_url(value):
    """Treat HTTP/HTTPS aliases as identical, without discarding query params."""
    url = urlsplit(value.strip())
    return urlunsplit(("https", url.netloc.lower(), url.path, url.query, ""))


def content_fingerprint(title, attributes):
    ordered = {key: sorted(values) for key, values in sorted(attributes.items())}
    return digest(title.casefold() + "\n" + json.dumps(ordered, ensure_ascii=False, sort_keys=True))


def is_laptop_replacement_keyboard(title: str, attributes: dict[str, list[str]]) -> bool:
    """识别装进特定笔记本型号的替换键盘，不排除普通外接键盘。"""
    laptop_keyboard_phrases = ("笔记本键盘", "笔记本电脑键盘")
    replacement_markers = ("适用于", "更换", "替换", "原装", "带框", "黑框", "内置", "维修")
    has_laptop_keyboard_phrase = any(phrase in title for phrase in laptop_keyboard_phrases)
    if has_laptop_keyboard_phrase and any(marker in title for marker in replacement_markers):
        return True

    # 此数据中笔记本内置键盘通常被标成 PS/2；普通外接键盘一般是 USB/蓝牙。
    interface_values = attributes.get("接口类型", [])
    has_internal_interface = any("ps/2" in value.casefold() for value in interface_values)
    if not has_internal_interface:
        return False
    if has_laptop_keyboard_phrase:
        return True

    # 部分维修件标题只罗列适配型号，并不写“笔记本”。出现下列词时按独立外接键盘保留。
    external_keyboard_markers = (
        "usb",
        "有线",
        "无线",
        "蓝牙",
        "外接",
        "外插",
        "台式",
        "办公",
        "游戏键盘",
        "机械键盘",
        "数字小键盘",
        "儿童键盘",
    )
    folded_title = title.casefold()
    return not any(marker in folded_title for marker in external_keyboard_markers)


def assign_splits(records, seed):
    """Exact 80/10/remainder allocation within each fine category, after dedup."""
    groups = defaultdict(list)
    for record in records:
        groups[record["category_l2"]].append(record)
    for group in groups.values():
        group.sort(key=lambda r: digest(f"split:{seed}:{r['product_id']}"))
        n_train, n_val = int(len(group) * 0.8), int(len(group) * 0.1)
        for index, record in enumerate(group):
            if index < n_train:
                record["split"] = "train"
            elif index < n_train + n_val:
                record["split"] = "validation"
            else:
                record["split"] = "test"
    return records


def build_dataset(config, root, output, config_path):
    """构建实验子集；磁盘候选库只在本次处理期间存在，结束后自动清理。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ecr-candidates-", dir=output.parent) as temporary:
        database_path = Path(temporary) / "candidates.sqlite"
        with closing(sqlite3.connect(database_path)) as connection:
            return _build_dataset(config, root, output, config_path, connection)


def _build_dataset(config, root, output, config_path, connection):
    """筛选、去重、抽样并写出结果；候选库连接由调用方管理。"""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Use a new --output directory.")
    source_paths = {key: (root / value).resolve() for key, value in config["sources"].items()}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    lookup = {}
    for category in config["categories"]:
        for label in category["raw_labels"]:
            if label in lookup:
                raise ValueError(f"Duplicate taxonomy rule: {label}")
            lookup[label] = category
    output.mkdir(parents=True)
    start = time.monotonic()
    counts, by_category, rejected = Counter(), Counter(), Counter()
    connection.execute(
        "CREATE TABLE candidates ("
        "id TEXT PRIMARY KEY, "
        "category TEXT, "
        "fingerprint TEXT UNIQUE, "
        "image_key TEXT UNIQUE, "
        "rank TEXT, "
        "payload TEXT"
        ")"
    )
    for product_id, raw in iter_products(source_paths["full"]):
        counts["source_records"] += 1
        if counts["source_records"] % 500000 == 0:
            connection.commit()
            print(
                f"Scanned {counts['source_records']:,}; "
                f"target records {counts['target_records']:,}",
                flush=True,
            )
        category = lookup.get(raw.get("label"))
        if category is None:
            continue
        counts["target_records"] += 1
        title, url, pv = raw.get("title"), raw.get("url"), raw.get("pv")
        if not all(isinstance(value, str) for value in (title, url, pv)):
            rejected["non_string_required_field"] += 1
            continue
        title = normalize_text(title)
        if not title or len(title) > config["max_title_chars"]:
            rejected["empty_or_oversized_title"] += 1
            continue
        if len(pv) > config["max_pv_chars"]:
            rejected["oversized_pv"] += 1
            continue
        try:
            parsed_url = urlsplit(url.strip())
            valid_url = (
                parsed_url.scheme in ("http", "https")
                and parsed_url.hostname in config["image_hosts"]
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            rejected["invalid_or_unsupported_image_url"] += 1
            continue
        attributes, issues = parse_pv(pv)
        if issues or len(attributes) < config["min_attribute_keys"]:
            rejected["malformed_or_insufficient_attributes"] += 1
            continue
        if any(word in title for word in category.get("exclude_title_terms", [])):
            rejected["excluded_accessory_or_bundle"] += 1
            continue
        if category.get(
            "exclude_laptop_replacement_keyboards"
        ) and is_laptop_replacement_keyboard(title, attributes):
            rejected["excluded_laptop_replacement_keyboard"] += 1
            continue
        excluded_attribute_values = category.get("exclude_attribute_values", {})
        if any(
            value in forbidden
            for key, forbidden in excluded_attribute_values.items()
            for value in attributes.get(key, [])
        ):
            rejected["excluded_use_case"] += 1
            continue
        fingerprint = content_fingerprint(title, attributes)
        image_key = digest(canonical_url(url))
        duplicate = connection.execute(
            "SELECT id, fingerprint, image_key FROM candidates WHERE id=? OR fingerprint=? OR image_key=? LIMIT 1",
            (product_id, fingerprint, image_key),
        ).fetchone()
        if duplicate:
            if duplicate[0] == product_id:
                reason = "duplicate_id"
            elif duplicate[1] == fingerprint:
                reason = "duplicate_content"
            else:
                reason = "duplicate_image_url"
            rejected[reason] += 1
            continue
        record = {
            "product_id": product_id,
            "title": title,
            "raw_label": raw["label"],
            "category_l1": category["category_l1"],
            "category_l2": category["category_l2"],
            "category_mapping": {
                "method": "exact_label_whitelist",
                "version": config["taxonomy_version"],
                "reviewed": False,
            },
            "attributes": attributes,
            "description": None,
            "image_url": url.strip(),
            "image_path": None,
            "image_status": "pending",
            "video_url": raw.get("video") or None,
            "raw": {"title": raw["title"], "pv": pv},
            "source_files": [source_paths["full"].name],
            "original_splits": [],
            "content_fingerprint": fingerprint,
            "image_url_fingerprint": image_key,
            "quality_flags": ["category_needs_review", "attributes_not_fact_checked"],
        }
        if any(len(values) > 1 for values in attributes.values()):
            record["quality_flags"].append("multivalue_attributes_need_selection")
        rank = digest(f"sample:{config['seed']}:{product_id}")
        connection.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?)",
            (
                product_id,
                category["category_l2"],
                fingerprint,
                image_key,
                rank,
                json.dumps(record, ensure_ascii=False),
            ),
        )
        by_category[category["category_l2"]] += 1
    connection.commit()
    selected = []
    for category in config["categories"]:
        rows = connection.execute(
            "SELECT payload FROM candidates WHERE category=? ORDER BY rank,id LIMIT ?",
            (category["category_l2"], config["per_category"]),
        )
        selected.extend(json.loads(row[0]) for row in rows)
    if not selected:
        raise ValueError("No candidates selected. Review the taxonomy and quality filters.")
    selected_by_id = {record["product_id"]: record for record in selected}
    source_records = {"full": counts["source_records"]}
    for split in ("train", "test"):
        n = 0
        for product_id, _ in iter_products(source_paths[split]):
            n += 1
            if product_id in selected_by_id:
                record = selected_by_id[product_id]
                if split not in record["original_splits"]:
                    record["original_splits"].append(split)
                    record["source_files"].append(source_paths[split].name)
            if n % 1000000 == 0:
                print(f"Tracing original {split}: {n:,}", flush=True)
        source_records[split] = n
    for record in selected:
        if len(record["original_splits"]) == 2:
            record["quality_flags"].append("original_train_test_overlap")
        elif not record["original_splits"]:
            record["quality_flags"].append("original_split_unknown")
    assign_splits(selected, config["seed"])
    selected.sort(key=lambda record: record["product_id"])
    for field in ("product_id", "content_fingerprint", "image_url_fingerprint"):
        if len({record[field] for record in selected}) != len(selected):
            raise AssertionError(f"Duplicate {field} in selected set")
    write_jsonl(output / "products.jsonl", selected)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", (r for r in selected if r["split"] == split))
    summary = {
        "dataset_version": config["dataset_version"],
        "config_sha256": digest(Path(config_path).read_text(encoding="utf-8")),
        "sources": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "records": source_records[key],
            }
            for key, path in source_paths.items()
        },
        "counts": dict(counts),
        "rejected": dict(rejected),
        "eligible_by_category": dict(by_category),
        "selected": len(selected),
        "selected_by_category": dict(Counter(r["category_l2"] for r in selected)),
        "splits": {
            split: dict(
                Counter(r["category_l2"] for r in selected if r["split"] == split)
            )
            for split in ("train", "validation", "test")
        },
        "selected_original_overlap": sum(len(r["original_splits"]) == 2 for r in selected),
        "selected_original_split_unknown": sum(not r["original_splits"] for r in selected),
        "duplicate_checks": {
            "product_id": 0,
            "content_fingerprint": 0,
            "image_url_fingerprint": 0,
        },
        "limitations": [
            "Provisional categories require review",
            "No human-written selling points/descriptions",
            "Images pending download",
            "No perceptual image or fuzzy text deduplication",
            "This experiment uses a new stratified split, not the original benchmark split",
        ],
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "config.json", config)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
