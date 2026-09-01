"""Pipeline step 5: top-k semantic search over the FAISS index.

Usage:
    python scripts/retrieval.py "скільки днів відпустки мені належить"
    python scripts/retrieval.py "звільнення під час лікарняного" -k 3
    python scripts/retrieval.py            # interactive prompt loop

Importable as a module:
    from retrieval import Retriever
    hits = Retriever().search("мінімальна заробітна плата", k=5)

Encoder settings (model name and the E5 prefixes) are read from
index/config.json, so the query can never be encoded differently from the
documents — the single most common way to silently break a retrieval layer.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"

DEFAULT_K = 5
PREVIEW_CHARS = 300


@dataclass
class Hit:
    rank: int
    chunk_id: str
    score: float
    text: str
    metadata: dict

    def preview(self, limit: int = PREVIEW_CHARS) -> str:
        """Chunk body without the breadcrumb header, trimmed for display."""
        body = self.text.split("\n\n", 1)[-1].replace("\n", " ")
        return body if len(body) <= limit else body[:limit].rstrip() + "…"

    @property
    def header(self) -> str:
        return self.text.split("\n\n", 1)[0]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "header": self.header,
            "preview": self.preview(),
            "metadata": self.metadata,
        }


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        for name in ("faiss.index", "meta.json", "config.json"):
            if not (index_dir / name).exists():
                raise SystemExit(
                    f"{index_dir / name} not found — run scripts/build_index.py first"
                )

        self.config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.model = SentenceTransformer(self.config["model"])

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        vector = self.model.encode(
            [self.config["query_prefix"] + query],
            normalize_embeddings=self.config["normalized"],
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, positions = self.index.search(vector, k)

        hits: list[Hit] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            entry = self.meta[position]
            hits.append(
                Hit(
                    rank=rank,
                    chunk_id=entry["chunk_id"],
                    score=float(score),
                    text=entry["text"],
                    metadata=entry["metadata"],
                )
            )
        return hits


def print_hits(query: str, hits: list[Hit]) -> None:
    print(f"\nQuery: {query}")
    print("=" * 78)
    for hit in hits:
        meta = hit.metadata
        print(f"\nTop-{hit.rank}: {hit.chunk_id} | score: {hit.score:.4f}")
        print(f"  Section: {meta.get('section')}")
        print(f"  Source:  {meta.get('source_file')} ({meta.get('document_id')})")
        print(f"  URL:     {meta.get('source_url')}")
        for line in textwrap.wrap(hit.preview(), width=74):
            print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the knowledge base")
    parser.add_argument("query", nargs="*", help="search query (omit for interactive mode)")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="number of results")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    retriever = Retriever()
    print(
        f"Model: {retriever.config['model']} | "
        f"{retriever.config['vectors']:,} vectors x {retriever.config['dimension']} dims"
    )

    if args.query:
        query = " ".join(args.query)
        hits = retriever.search(query, k=args.k)
        if args.json:
            print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
        else:
            print_hits(query, hits)
        return

    print("Interactive mode — empty line or Ctrl+C to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        print_hits(query, retriever.search(query, k=args.k))


if __name__ == "__main__":
    sys.exit(main())
