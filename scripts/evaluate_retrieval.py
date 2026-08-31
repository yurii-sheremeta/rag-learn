"""Quantitative check: at which rank does the correct article actually appear?

The qualitative report in outputs/retrieval_examples.md shows what came back;
this script answers the sharper question — for each test query, the article of
law that genuinely answers it is known in advance, so we can measure recall@k
instead of eyeballing the output.

Ground truth is (document_id, article) rather than a chunk_id: an article is
usually split across several chunks, and any of them counts as a hit.

Usage:  python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "evaluation.json"
MAX_K = 10

# query -> the article that actually answers it
GROUND_TRUTH: list[dict[str, str]] = [
    {
        "query": "Скільки днів щорічної відпустки мені належить?",
        "document_id": "zakon_pro_vidpustky",
        "article": "Стаття 6.",
    },
    {
        "query": "Чи можуть звільнити працівника під час лікарняного?",
        "document_id": "kzpp",
        "article": "Стаття 40.",
    },
    {
        "query": "Коли роботодавець має виплатити розрахунок при звільненні?",
        "document_id": "kzpp",
        "article": "Стаття 116.",
    },
    {
        "query": "Чи можна встановити випробувальний термін під час воєнного стану?",
        "document_id": "zakon_pro_voiennyi_stan",
        "article": "Стаття 2.",
    },
    {
        "query": "Хто розслідує нещасний випадок на виробництві?",
        "document_id": "zakon_pro_ohoronu_praci",
        "article": "Стаття 22.",
    },
    {
        "query": "Що таке мінімальна заробітна плата?",
        "document_id": "zakon_pro_oplatu_praci",
        "article": "Стаття 3.",
    },
    {
        "query": "Скільки годин на тиждень я маю працювати?",
        "document_id": "kzpp",
        "article": "Стаття 50.",
    },
    {
        "query": "Мене змушують вийти на роботу у вихідний, чи це законно?",
        "document_id": "kzpp",
        "article": "Стаття 71.",
    },
    {
        "query": "Чи можу я взяти відпустку за свій рахунок на два тижні?",
        "document_id": "zakon_pro_vidpustky",
        "article": "Стаття 25.",
    },
    {
        "query": "Що робити, якщо роботодавець не виконує колективний договір?",
        "document_id": "zakon_pro_kolektyvni_dohovory",
        "article": "Стаття 18.",
    },
]


def main() -> None:
    retriever = Retriever()
    results: list[dict] = []

    for item in GROUND_TRUTH:
        hits = retriever.search(item["query"], k=MAX_K)
        rank = None
        for hit in hits:
            metadata = hit.metadata
            if metadata["document_id"] == item["document_id"] and str(
                metadata.get("section", "")
            ).startswith(item["article"]):
                rank = hit.rank
                break

        results.append({**item, "rank": rank, "top1_chunk_id": hits[0].chunk_id})
        label = f"rank {rank}" if rank else f"not in top-{MAX_K}"
        print(f"{label:>16} | {item['article']:12s} | {item['query'][:48]}")

    total = len(results)
    recall = {
        f"recall@{k}": sum(1 for r in results if r["rank"] and r["rank"] <= k) / total
        for k in (1, 3, 5, MAX_K)
    }

    print()
    for name, value in recall.items():
        print(f"{name}: {value:.0%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {"max_k": MAX_K, "queries": total, "recall": recall, "results": results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
