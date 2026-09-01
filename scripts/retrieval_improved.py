"""Improved retrieval: metadata filtering + cross-encoder reranking.

The HW2 baseline measured recall@1 = 50% against recall@3 = 90%: the correct
article is almost always retrieved, just not ranked first. That gap is what
this module closes, in four stages.

    1. over-fetch     bi-encoder returns CANDIDATE_K candidates, not k
    2. metadata filter  narrow by article number / act named in the query
    3. dedup          drop cross-act near-copies, keep the special law
    4. rerank         cross-encoder reorders what survived, return top k

Every stage can be switched off independently so their contributions can be
measured separately — see scripts/compare_retrieval.py.

Why a cross-encoder is stronger than the bi-encoder it reranks: the bi-encoder
embeds query and document separately and can only compare the two finished
vectors, so nothing in the document ever "sees" the query. The cross-encoder
reads the pair together in one pass and scores their relation directly. That
is far more accurate and far too slow to run over 763 chunks — which is
exactly why it runs over 20 candidates instead.

Usage:
    python scripts/retrieval_improved.py "скільки годин на тиждень я маю працювати"
    python scripts/retrieval_improved.py "стаття 116" --explain
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from retrieval import INDEX_DIR, Hit, Retriever


CANDIDATE_K = 20
DEFAULT_K = 3
DEDUP_THRESHOLD = 0.92
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DOCUMENT_TYPE_PRIORITY = {"law": 0, "code": 1}


ARTICLE_IN_QUERY_RE = re.compile(
    r"\b(?:стат[тьіюя]\w*|ст\.?)\s*(\d+(?:-\d+)?)", re.IGNORECASE
)

ACT_ROUTING: dict[str, str] = {
    "кзпп": "kzpp",
    "кодекс законів про працю": "kzpp",
    "про відпустки": "zakon_pro_vidpustky",
    "про оплату праці": "zakon_pro_oplatu_praci",
    "про охорону праці": "zakon_pro_ohoronu_praci",
    "про колективні договори": "zakon_pro_kolektyvni_dohovory",
    "воєнного стану": "zakon_pro_voiennyi_stan",
    "воєнний стан": "zakon_pro_voiennyi_stan",
}


@dataclass
class QueryFilters:
    """Metadata constraints, either detected in the query or passed by hand."""

    article_no: str | None = None
    document_ids: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.article_no or self.document_ids or self.document_types)

    def describe(self) -> str:
        parts = []
        if self.article_no:
            parts.append(f"article_no={self.article_no}")
        if self.document_ids:
            parts.append(f"document_id in {self.document_ids}")
        if self.document_types:
            parts.append(f"document_type in {self.document_types}")
        return ", ".join(parts) if parts else "—"

    def matches(self, metadata: dict) -> bool:
        if self.article_no and metadata.get("article_no") != self.article_no:
            return False
        if self.document_ids and metadata.get("document_id") not in self.document_ids:
            return False
        if (
            self.document_types
            and metadata.get("document_type") not in self.document_types
        ):
            return False
        return True


def detect_filters(query: str) -> QueryFilters:
    """Read metadata constraints out of the wording of the query."""
    filters = QueryFilters()

    article = ARTICLE_IN_QUERY_RE.search(query)
    if article:
        filters.article_no = article.group(1)

    lowered = query.lower()
    for phrase, document_id in ACT_ROUTING.items():
        if phrase in lowered and document_id not in filters.document_ids:
            filters.document_ids.append(document_id)

    return filters


@dataclass
class RankedHit(Hit):
    """A Hit that remembers where it came from and what happened to it."""

    base_rank: int = 0
    base_score: float = 0.0
    rerank_score: float | None = None
    absorbed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload.update(
            {
                "base_rank": self.base_rank,
                "base_score": round(self.base_score, 4),
                "rerank_score": (
                    None if self.rerank_score is None else round(self.rerank_score, 4)
                ),
                "absorbed_duplicates": self.absorbed,
            }
        )
        return payload


class ImprovedRetriever:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        reranker_model: str = RERANKER_MODEL,
        load_reranker: bool = True,
    ) -> None:
        self.base = Retriever(index_dir)
        self.reranker_name = reranker_model
        self._reranker = None

        self.vectors = np.vstack(
            [self.base.index.reconstruct(i) for i in range(self.base.index.ntotal)]
        )
        self.position_of = {
            entry["chunk_id"]: i for i, entry in enumerate(self.base.meta)
        }

        if load_reranker:
            self._load_reranker()

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_name)
        return self._reranker


    def _apply_filters(
        self, hits: list[RankedHit], filters: QueryFilters
    ) -> list[RankedHit]:
        if filters.is_empty():
            return hits
        kept = [hit for hit in hits if filters.matches(hit.metadata)]
        return kept if kept else hits


    def _deduplicate(self, hits: list[RankedHit]) -> list[RankedHit]:
        """Drop near-identical chunks that come from different acts.

        Threshold alone cannot do this job. Measured on our corpus:
        adjacent chunks of the SAME article sit at ~0.967 similarity, while
        genuine cross-act duplicates sit at ~0.944 — the duplicates are less
        similar than the pairs we want to keep. So similarity only decides
        among chunks whose `document_id` differs; inside one document nothing
        is ever dropped.
        """
        survivors: list[RankedHit] = []

        for hit in hits:
            vector = self.vectors[self.position_of[hit.chunk_id]]
            duplicate_of = None

            for kept in survivors:
                if kept.metadata["document_id"] == hit.metadata["document_id"]:
                    continue
                similarity = float(vector @ self.vectors[self.position_of[kept.chunk_id]])
                if similarity >= DEDUP_THRESHOLD:
                    duplicate_of = kept
                    break

            if duplicate_of is None:
                survivors.append(hit)
                continue

            incoming = DOCUMENT_TYPE_PRIORITY.get(hit.metadata["document_type"], 9)
            existing = DOCUMENT_TYPE_PRIORITY.get(
                duplicate_of.metadata["document_type"], 9
            )
            if incoming < existing:
                position = survivors.index(duplicate_of)
                hit.absorbed = duplicate_of.absorbed + [duplicate_of.chunk_id]
                survivors[position] = hit
            else:
                duplicate_of.absorbed.append(hit.chunk_id)

        return survivors


    def _rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        if not hits:
            return hits
        reranker = self._load_reranker()
        pairs = [(query, hit.text) for hit in hits]
        raw = np.asarray(reranker.predict(pairs), dtype=np.float64)
        if raw.min() < 0.0 or raw.max() > 1.0:
            raw = 1.0 / (1.0 + np.exp(-raw))
        probabilities = raw

        for hit, value in zip(hits, probabilities):
            hit.rerank_score = float(value)
            hit.score = float(value)

        return sorted(hits, key=lambda hit: hit.rerank_score or 0.0, reverse=True)


    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        use_filters: bool = True,
        use_dedup: bool = True,
        use_rerank: bool = True,
        filters: QueryFilters | None = None,
    ) -> list[RankedHit]:
        candidates = [
            RankedHit(
                rank=hit.rank,
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
                base_rank=hit.rank,
                base_score=hit.score,
            )
            for hit in self.base.search(query, k=CANDIDATE_K)
        ]

        if use_filters:
            active = filters if filters is not None else detect_filters(query)
            candidates = self._apply_filters(candidates, active)
        elif filters is not None:
            candidates = self._apply_filters(candidates, filters)

        if use_dedup:
            candidates = self._deduplicate(candidates)

        if use_rerank:
            candidates = self._rerank(query, candidates)

        for rank, hit in enumerate(candidates[:k], start=1):
            hit.rank = rank
        return candidates[:k]


def print_hits(query: str, hits: list[RankedHit], filters: QueryFilters) -> None:
    print(f"\nQuery: {query}")
    print(f"Filters: {filters.describe()}")
    print("=" * 78)
    for hit in hits:
        meta = hit.metadata
        moved = hit.base_rank - hit.rank
        arrow = f"  (was #{hit.base_rank}{', +' + str(moved) if moved > 0 else ''})"
        print(f"\nTop-{hit.rank}: {hit.chunk_id}{arrow}")
        print(
            f"  rerank: {hit.rerank_score:.4f}"
            if hit.rerank_score is not None
            else f"  score: {hit.score:.4f}"
        )
        print(f"  Section: {meta.get('section')}")
        print(f"  Source:  {meta.get('source_file')} ({meta.get('document_id')})")
        if hit.absorbed:
            print(f"  Absorbed duplicates: {', '.join(hit.absorbed)}")
        for line in textwrap.wrap(hit.preview(), width=74):
            print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved retrieval pipeline")
    parser.add_argument("query", nargs="*")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--no-filters", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--document-type", help="force a document_type filter")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    retriever = ImprovedRetriever(load_reranker=not args.no_rerank)

    if not args.query:
        parser.error("provide a query")

    query = " ".join(args.query)
    manual = None
    if args.document_type:
        manual = detect_filters(query)
        manual.document_types = [args.document_type]

    hits = retriever.search(
        query,
        k=args.k,
        use_filters=not args.no_filters,
        use_dedup=not args.no_dedup,
        use_rerank=not args.no_rerank,
        filters=manual,
    )

    if args.json:
        print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
    else:
        print_hits(query, hits, manual or detect_filters(query))


if __name__ == "__main__":
    sys.exit(main())
