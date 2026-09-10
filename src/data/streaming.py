"""Read a top-level JSON object incrementally, independent of indentation."""

import json
from pathlib import Path


def iter_products(path, chunk_size=1024 * 1024, max_record_chars=4 * 1024 * 1024):
    """Yield (ID, record) without loading the full JSON document into memory.

    Reject truncated input, trailing commas/content and oversized individual
    values. Duplicate IDs are yielded so downstream code can audit them.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as handle:
        buffer, pos, eof = "", 0, False

        def refill():
            nonlocal buffer, pos, eof
            data = handle.read(chunk_size)
            buffer = buffer[pos:] + data
            pos = 0
            eof = not data

        def whitespace():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in " \r\n\t":
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                refill()

        def expect(char):
            nonlocal pos
            whitespace()
            if pos >= len(buffer) or buffer[pos] != char:
                raise ValueError(f"{path}: expected {char!r}, near {buffer[pos:pos+60]!r}")
            pos += 1

        def value():
            nonlocal pos
            whitespace()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError as error:
                    if eof or len(buffer) - pos > max_record_chars:
                        raise ValueError(f"{path}: invalid/truncated/oversized JSON value") from error
                    refill()
                    continue
                if end - pos > max_record_chars:
                    raise ValueError(f"{path}: oversized JSON value")
                pos = end
                return result

        expect("{")
        whitespace()
        if pos < len(buffer) and buffer[pos] == "}":
            pos += 1
        else:
            while True:
                key = value()
                if not isinstance(key, str):
                    raise ValueError(f"{path}: product ID must be a string")
                expect(":")
                record = value()
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: product {key} must be an object")
                yield key, record
                whitespace()
                if pos < len(buffer) and buffer[pos] == "}":
                    pos += 1
                    break
                expect(",")
        whitespace()
        if pos < len(buffer):
            raise ValueError(f"{path}: trailing content after JSON object")
