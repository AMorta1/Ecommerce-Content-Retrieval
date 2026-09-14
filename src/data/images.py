"""Download a bounded local image subset, validate bytes, and build ready splits."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
import urllib.request
from urllib.parse import urlsplit
import warnings

from PIL import Image

from .preprocess import canonical_url, write_json, write_jsonl

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 24_000_000
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


def validate_image(data):
    """Check full decode as well as file integrity, using actual image format."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            fmt = image.format
            if width < 64 or height < 64 or width * height > MAX_PIXELS:
                raise ValueError(f"Unsupported image dimensions: {width}x{height}")
            if fmt not in FORMATS:
                raise ValueError(f"Unsupported image format: {fmt}")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
    return {"width": width, "height": height, "format": fmt,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class AllowedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts):
        super().__init__()
        self.hosts = set(hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urlsplit(newurl)
        if url.scheme != "https" or url.hostname not in self.hosts:
            raise ValueError(f"Unsupported redirect target: {url.hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_image(record, root, image_dir, hosts, timeout=12, retries=1):
    """先校验本版本的缓存；缓存不可用时下载，返回路径或明确的失败原因。"""
    rid = record["product_id"]
    if not re.fullmatch(r"[0-9]+", rid):
        raise ValueError(f"Unsafe product ID for image filename: {rid!r}")
    result = {"product_id": rid, "original_url": record["image_url"]}
    url = canonical_url(record["image_url"])
    if urlsplit(url).hostname not in hosts:
        return {**result, "status": "failed", "error": "unsupported_host"}
    for suffix in FORMATS.values():
        cached = image_dir / (rid + suffix)
        if cached.is_file() and cached.stat().st_size <= MAX_IMAGE_BYTES:
            try:
                metadata = validate_image(cached.read_bytes())
                return {
                    **result,
                    **metadata,
                    "status": "cached",
                    "image_path": cached.relative_to(root).as_posix(),
                }
            except (
                OSError,
                ValueError,
                SyntaxError,
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
            ):
                # 损坏的缓存不能直接使用，继续尝试下载。
                continue
    for attempt in range(retries + 1):
        try:
            opener = urllib.request.build_opener(AllowedRedirects(hosts))
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://www.taobao.com/",
                },
            )
            with opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
                content_type = response.headers.get("Content-Type")
                final_url = response.geturl()
                status = response.status
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image_exceeds_12_MiB")
            if status != 200:
                raise ValueError(f"Expected full image, got HTTP {status}")
            metadata = validate_image(data)
            target = image_dir / (rid + FORMATS[metadata["format"]])
            temporary = image_dir / (rid + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
            return {
                **result,
                **metadata,
                "status": "downloaded",
                "http_status": status,
                "content_type": content_type,
                "fetched_url": final_url,
                "image_path": target.relative_to(root).as_posix(),
            }
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(0.5)
    return {**result, "status": "failed"}


def smoke_examples(records, limit=5):
    """Use training records only; cover both broad and several fine categories."""
    buckets = defaultdict(dict)

    def example_rank(record):
        # Prefer modest-size, relatively unambiguous attribute sets for smoke tests.
        attrs = record["attributes"]
        return (
            sum(len(values) - 1 for values in attrs.values()),
            abs(len(attrs) - 10),
            record["product_id"],
        )

    for record in sorted(records, key=example_rank):
        if record["split"] == "train":
            buckets[record["category_l1"]].setdefault(record["category_l2"], record)
    queues = [list(categories.values()) for _, categories in sorted(buckets.items())]
    selected = []
    while any(queues) and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def download_dataset(root, dataset, workers=4, limit=None):
    """导出图片可用的数据；指定的推理样本由单独入口生成，避免重复维护。"""
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    with (dataset / "products.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if limit is not None:
        records = smoke_examples(records, limit)
    image_dir = root / "data/images" / dataset.name
    image_dir.mkdir(parents=True, exist_ok=True)
    output = dataset / ("multimodal" if limit is None else "smoke")
    output.mkdir(exist_ok=True)
    completed = []
    start = time.monotonic()
    with (output / "image_manifest.jsonl").open("w", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(fetch_image, record, root, image_dir, config["image_hosts"])
                for record in records
            ]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                if len(completed) % 100 == 0 or len(completed) == len(records):
                    failed = sum(r["status"] == "failed" for r in completed)
                    elapsed = time.monotonic() - start
                    print(
                        f"Images {len(completed)}/{len(records)}; "
                        f"failed {failed}; {elapsed:.1f}s",
                        flush=True,
                    )
    lookup = {record["product_id"]: record for record in completed}
    ready, seen_hashes, duplicates = [], {}, []
    for record in sorted(records, key=lambda r: r["product_id"]):
        image = lookup[record["product_id"]]
        if image["status"] == "failed":
            continue
        if image["sha256"] in seen_hashes:
            duplicates.append(
                {
                    "product_id": record["product_id"],
                    "retained_id": seen_hashes[image["sha256"]],
                }
            )
            continue
        seen_hashes[image["sha256"]] = record["product_id"]
        record.update(
            image_path=image["image_path"],
            image_status="validated",
            image_sha256=image["sha256"],
            image_width=image["width"],
            image_height=image["height"],
        )
        ready.append(record)
    write_jsonl(output / "products.jsonl", ready)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", (r for r in ready if r["split"] == split))
    summary = {
        "attempted": len(records),
        "image_status_counts": dict(Counter(r["status"] for r in completed)),
        "ready_records": len(ready),
        "exact_image_duplicates_removed": duplicates,
        "by_category": dict(Counter(r["category_l2"] for r in ready)),
        "splits": {
            split: dict(Counter(r["category_l2"] for r in ready if r["split"] == split))
            for split in ("train", "validation", "test")
        },
        "downloaded_or_cached_bytes": sum(r.get("bytes", 0) for r in completed),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "note": (
            "Original experiment split assignments retained after image filtering; "
            "counts may differ from 80/10/10."
        ),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not ready:
        raise RuntimeError("No valid images; inspect image_manifest.jsonl before continuing.")
    return summary
