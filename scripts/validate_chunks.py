"""Quality checks for the finished knowledge base.

Reads data/processed/chunks.jsonl and verifies:
  * the file is valid JSONL and every chunk_id is unique;
  * all required metadata fields are present;
  * chunk sizes stay within the configured bounds;
  * declared overlap is real (a chunk's opening text is in the previous one);
  * no HTML markup or editorial notes survived normalization.

Usage:  python scripts/validate_chunks.py
Exits with code 1 when critical errors are found.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

REQUIRED_TOP_LEVEL = ("chunk_id", "text", "metadata")
REQUIRED_METADATA = (
    "document_id",
    "source_file",
    "chunk_index",
    "title",
    "section",
    "language",
    "domain",
    "document_type",
)

MAX_CHARS = 1000
MIN_CHARS = 500
MARKUP_RE = re.compile(r"<[a-zA-Z/]|&[a-z]+;|\{|\}")


def main() -> int:
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found")
        return 1

    records: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    for line_number, line in enumerate(
        CHUNKS_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON — {exc}")

    if not records:
        print("ERROR: file is empty")
        return 1

    for record in records:
        for field in REQUIRED_TOP_LEVEL:
            if field not in record:
                errors.append(f"{record.get('chunk_id', '?')}: missing field {field}")
        metadata = record.get("metadata", {})
        for field in REQUIRED_METADATA:
            if field not in metadata:
                errors.append(
                    f"{record.get('chunk_id', '?')}: missing metadata.{field}"
                )

    duplicates = [
        chunk_id
        for chunk_id, count in Counter(r["chunk_id"] for r in records).items()
        if count > 1
    ]
    if duplicates:
        errors.append(f"duplicate chunk_id values: {duplicates[:5]}")

    by_document: dict[str, list[dict]] = {}
    for record in records:
        by_document.setdefault(record["metadata"]["document_id"], []).append(record)

    for document_id, document_records in by_document.items():
        indexes = [r["metadata"]["chunk_index"] for r in document_records]
        if indexes != list(range(len(document_records))):
            errors.append(f"{document_id}: chunk_index is not a contiguous sequence")

    sizes = [len(r["text"]) for r in records]
    oversized = [r["chunk_id"] for r in records if len(r["text"]) > MAX_CHARS]
    if oversized:
        errors.append(f"chunks longer than {MAX_CHARS} chars: {len(oversized)}")

    undersized = [r for r in records if len(r["text"]) < MIN_CHARS]
    if undersized:
        warnings.append(
            f"chunks shorter than {MIN_CHARS} chars: {len(undersized)} "
            f"({len(undersized) / len(records):.1%}) — whole short articles"
        )

    dirty = [r["chunk_id"] for r in records if MARKUP_RE.search(r["text"])]
    if dirty:
        errors.append(
            f"leftover markup or editorial notes in {len(dirty)} chunks: {dirty[:3]}"
        )

    overlap_declared = 0
    overlap_confirmed = 0
    for index, record in enumerate(records):
        if not record["metadata"].get("has_overlap_with_previous"):
            continue
        overlap_declared += 1
        body = record["text"].split("\n\n", 1)[-1]
        if index and body[:60] in records[index - 1]["text"]:
            overlap_confirmed += 1

    if overlap_declared and overlap_confirmed / overlap_declared < 0.9:
        errors.append(
            f"overlap confirmed for only {overlap_confirmed}/{overlap_declared}"
        )

    print("=" * 62)
    print("KNOWLEDGE BASE VALIDATION")
    print("=" * 62)
    print(f"Chunks:            {len(records):,}")
    print(f"Documents:         {len(by_document)}")
    print(f"Size, chars:       min={min(sizes)}  max={max(sizes)}  "
          f"avg={statistics.mean(sizes):.0f}  median={statistics.median(sizes):.0f}")
    print(f"Within 500-1000:   {sum(MIN_CHARS <= s <= MAX_CHARS for s in sizes)} "
          f"({sum(MIN_CHARS <= s <= MAX_CHARS for s in sizes) / len(sizes):.1%})")
    print(f"With overlap:      {overlap_declared} "
          f"(confirmed {overlap_confirmed})")
    print(f"Distinct articles: {len({r['metadata']['section'] for r in records})}")
    print("-" * 62)

    for warning in warnings:
        print(f"[warn]  {warning}")
    for error in errors:
        print(f"[ERROR] {error}")

    if errors:
        print("-" * 62)
        print(f"RESULT: {len(errors)} error(s) found")
        return 1

    print("-" * 62)
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
