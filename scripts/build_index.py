"""Pipeline step 4: embed the chunks and build the FAISS vector index.

    data/processed/chunks.jsonl
        -> embeddings           multilingual-e5-base, one vector per chunk
    index/faiss.index           FAISS IndexFlatIP (exact search)
    index/meta.json             row -> chunk_id + text + metadata
    index/config.json           model, dimension, prefixes — the retriever
                                reads these so query and documents can never
                                be encoded with different settings

Model choice: the assignment suggests all-MiniLM-L6-v2, which is English-only
and performs poorly on Ukrainian legal text. intfloat/multilingual-e5-base is
trained for multilingual retrieval and is used instead. E5 models require the
"passage: " / "query: " prefixes — omitting them measurably degrades quality.

Usage:  python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"

MODEL_NAME = "intfloat/multilingual-e5-base"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(
            f"{CHUNKS_PATH} not found — run scripts/prepare_knowledge_base.py first"
        )
    return [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    records = load_chunks()
    print(f"Loaded {len(records):,} chunks from {CHUNKS_PATH.relative_to(ROOT)}")

    print(f"Loading model {args.model} (first run downloads it)...")
    model = SentenceTransformer(args.model)

    texts = [DOCUMENT_PREFIX + record["text"] for record in records]
    print(f"Encoding {len(texts):,} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> inner product == cosine
        convert_to_numpy=True,
    ).astype(np.float32)

    dimension = int(embeddings.shape[1])
    # Exact search: 763 vectors make approximate indexes (IVF/HNSW) pointless.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    meta = [
        {
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "metadata": record["metadata"],
        }
        for record in records
    ]
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "model": args.model,
        "dimension": dimension,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "normalized": True,
        "index_type": "IndexFlatIP",
        "similarity": "cosine (inner product on unit vectors)",
        "vectors": int(index.ntotal),
        "built_at": date.today().isoformat(),
    }
    (INDEX_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"\nIndex built: {index.ntotal:,} vectors x {dimension} dims -> "
        f"{(INDEX_DIR / 'faiss.index').relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
