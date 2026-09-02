"""Retrieve context for every test question once and cache it to disk.

Prompt iteration changes the instruction, never the retrieved chunks. Running
the cross-encoder again for each prompt revision would repeat the most
expensive part of the pipeline for no benefit, so retrieval runs once here.

The cache is written after every question, so a killed process (the reranker
has been OOM-killed on this machine) loses at most one question: rerunning
picks up where it stopped.

Usage:
    python scripts/build_retrieval_cache.py
    python scripts/build_retrieval_cache.py --no-rerank
    python scripts/build_retrieval_cache.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from qa_questions import TEST_QUESTIONS  # noqa: E402
from retrieval_improved import ImprovedRetriever, detect_filters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "outputs" / "retrieval_cache.json"

TOP_K = 3


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"built_at": date.today().isoformat(), "top_k": TOP_K, "questions": {}}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="ignore existing cache")
    args = parser.parse_args()

    cache = {"built_at": date.today().isoformat(), "top_k": TOP_K, "questions": {}}
    if not args.rebuild:
        cache = load_cache()

    pending = [q for q in TEST_QUESTIONS if q["id"] not in cache["questions"]]
    if not pending:
        print(f"Cache already complete: {len(cache['questions'])} questions")
        return

    print(f"{len(cache['questions'])} cached, {len(pending)} to retrieve")
    retriever = ImprovedRetriever(load_reranker=not args.no_rerank)
    cache["reranked"] = not args.no_rerank
    cache["model"] = retriever.base.config["model"]

    for item in pending:
        hits = retriever.search(
            item["question"], k=TOP_K, use_rerank=not args.no_rerank
        )
        filters = detect_filters(item["question"])
        cache["questions"][item["id"]] = {
            "question": item["question"],
            "kind": item["kind"],
            "filters": filters.describe(),
            "hits": [hit.to_dict() for hit in hits],
            "chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 4),
                    "text": hit.text,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
        }
        save_cache(cache)
        top = hits[0]
        print(
            f"[ok] {item['id']:32s} {top.chunk_id:34s} {top.score:.3f} "
            f"{top.metadata['section'][:42]}"
        )

    print(f"\nWrote {CACHE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
