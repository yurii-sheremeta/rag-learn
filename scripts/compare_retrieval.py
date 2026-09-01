"""Baseline vs improved retrieval on the same ten queries.

Adding a technique is easy; showing it helped is the actual task. This script
runs the HW2 baseline and the HW3 pipeline over one query set and one set of
known-correct articles, then reports recall@k for each.

It also runs an ablation — filters only, dedup only, rerank only — because
"the pipeline got better" is a weaker claim than "this stage is what moved
the number".

Output: outputs/retrieval_comparison.md and outputs/comparison.json

Usage:  python scripts/compare_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluate_retrieval import GROUND_TRUTH
from retrieval_improved import ImprovedRetriever, detect_filters

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "outputs" / "retrieval_comparison.md"
OUTPUT_JSON = ROOT / "outputs" / "comparison.json"

TOP_K = 3

CONFIGURATIONS: dict[str, dict[str, bool]] = {
    "baseline": {"use_filters": False, "use_dedup": False, "use_rerank": False},
    "+ filters": {"use_filters": True, "use_dedup": False, "use_rerank": False},
    "+ dedup": {"use_filters": False, "use_dedup": True, "use_rerank": False},
    "+ rerank": {"use_filters": False, "use_dedup": False, "use_rerank": True},
    "improved": {"use_filters": True, "use_dedup": True, "use_rerank": True},
}


FILTER_DEMOS: list[dict[str, str]] = [
    {
        "query": "що каже стаття 116 про строки розрахунку",
        "expect": "article_no фільтрує 20 кандидатів до чанків однієї статті",
    },
    {
        "query": "які обов'язки роботодавця встановлює закон про охорону праці",
        "expect": "document_id маршрутизує пошук в один акт",
    },
    {
        "query": "стаття 40 КЗпП підстави звільнення",
        "expect": "обидва фільтри одночасно — номер статті і акт",
    },
]


def is_gold(metadata: dict, item: dict) -> bool:
    """Does this chunk belong to the article that answers the query?"""
    return metadata["document_id"] == item["document_id"] and str(
        metadata.get("section", "")
    ).startswith(item["article"])


def gold_rank(hits, item: dict) -> int | None:
    for hit in hits:
        if is_gold(hit.metadata, item):
            return hit.rank
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for rank in ranks if rank and rank <= k) / len(ranks)


def explain(baseline_hit, improved_hit, filters, item, base_rank, new_rank) -> str:
    """A factual sentence about what the pipeline did to this query.

    Judged by the article that reached top-1, not by chunk id: swapping one
    fragment of the correct article for another is not an improvement, and
    reporting it as one would overstate the result.
    """
    reasons = []
    if not filters.is_empty():
        reasons.append(f"фільтр {filters.describe()}")
    if improved_hit.absorbed:
        reasons.append("дедуплікація прибрала " + ", ".join(improved_hit.absorbed))
    if improved_hit.base_rank > 1:
        reasons.append(f"reranker підняв із позиції #{improved_hit.base_rank}")
    detail = ("; ".join(reasons)) if reasons else "стадії не спрацювали"

    baseline_correct = base_rank == 1
    improved_correct = new_rank == 1

    if improved_correct and not baseline_correct:
        return f"**виправлено** — потрібна стаття стала top-1: {detail}"
    if baseline_correct and not improved_correct:
        return f"**регресія** — правильна стаття втратила top-1: {detail}"
    if baseline_correct and improved_correct:
        if baseline_hit.chunk_id == improved_hit.chunk_id:
            return "без змін — baseline уже був точним"
        return (
            "та сама стаття лишилась top-1, змінився лише фрагмент: " + detail
        )
    return f"не виправлено — правильної статті немає в top-1 в обох варіантах: {detail}"


def render(payload: dict) -> str:
    lines = [
        "# Порівняння: baseline vs improved retrieval",
        "",
        f"Згенеровано: {payload['generated_at']} · "
        "відтворюється командою `python scripts/compare_retrieval.py`",
        "",
        "**Baseline** — пайплайн ДЗ №2: bi-encoder `multilingual-e5-base` + "
        "`IndexFlatIP`, top-3 напряму.",
        "",
        "**Improved** — той самий bi-encoder дістає 20 кандидатів, далі "
        "metadata-фільтри, дедуплікація між актами і cross-encoder "
        "`BAAI/bge-reranker-v2-m3`, який переставляє те, що лишилось.",
        "",
        "---",
        "",
        "## Внесок кожної стадії",
        "",
        "Ті самі 10 запитів, той самий набір правильних статей. "
        "Кожна стадія вмикається окремо, щоб було видно, яка з них рухає метрику.",
        "",
        "| Конфігурація | recall@1 | recall@3 | Δ recall@1 |",
        "|---|---|---|---|",
    ]

    base_r1 = payload["configurations"]["baseline"]["recall@1"]
    for name, stats in payload["configurations"].items():
        delta = stats["recall@1"] - base_r1
        delta_text = "—" if name == "baseline" else f"{delta:+.0%}"
        lines.append(
            f"| {name} | {stats['recall@1']:.0%} | {stats['recall@3']:.0%} | {delta_text} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Порівняння по запитах",
        "",
        "| # | Query | Baseline top-1 | Improved top-1 | Що змінилось |",
        "|---|---|---|---|---|",
    ]

    for number, row in enumerate(payload["queries"], start=1):
        lines.append(
            f"| {number} | {row['query']} | `{row['baseline_top1']}` | "
            f"`{row['improved_top1']}` | {row['explanation']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Metadata filtering окремо",
        "",
        "З десяти оцінювальних запитів номер статті не називає жоден, а акт — "
        "лише запит №4 («під час воєнного стану»), тож там із метаданих "
        "переважно працює дедуплікація. Ці три запити показують фільтри в дії "
        "й винесені окремо, щоб не впливати на цифри вище.",
        "",
        "| Query | Виявлені фільтри | Кандидатів до → після | Top-1 без фільтра | Top-1 з фільтром |",
        "|---|---|---|---|---|",
    ]

    for demo in payload["filter_demos"]:
        lines.append(
            f"| {demo['query']} | `{demo['filters']}` | "
            f"{demo['candidates_before']} → {demo['candidates_after']} | "
            f"`{demo['top1_without']}` | `{demo['top1_with']}` |"
        )

    lines += ["", "---", "", "## Деталі по кожному запиту", ""]

    for number, row in enumerate(payload["queries"], start=1):
        lines.append(f"### {number}. {row['query']}")
        lines.append("")
        lines.append(f"**Очікувана стаття:** {row['article']} ({row['document_id']})")
        lines.append("")
        lines.append(f"**Виявлені фільтри:** {row['filters']}")
        lines.append("")
        lines.append("| | Baseline | Improved |")
        lines.append("|---|---|---|")
        lines.append(
            f"| top-1 | `{row['baseline_top1']}` | `{row['improved_top1']}` |"
        )
        lines.append(
            f"| секція top-1 | {row['baseline_section']} | {row['improved_section']} |"
        )
        lines.append(
            f"| позиція правильної статті | "
            f"{row['baseline_gold_rank'] or 'не в top-3'} | "
            f"{row['improved_gold_rank'] or 'не в top-3'} |"
        )
        lines.append("")
        lines.append(row["explanation"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-ablation",
        action="store_true",
        help=(
            "read the five-configuration table from a previous "
            "outputs/comparison.json instead of recomputing it. The ablation "
            "is two thirds of the cross-encoder work, so this is the flag to "
            "use when only the per-query report changed."
        ),
    )
    args = parser.parse_args()

    improved_retriever = ImprovedRetriever()
    baseline_retriever = improved_retriever.base

    configurations: dict[str, dict] = {}

    if args.reuse_ablation and OUTPUT_JSON.exists():
        cached = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        configurations = cached["configurations"]
        print("[cache] ablation table reused from outputs/comparison.json")
        for name, stats in configurations.items():
            print(
                f"[ok] {name:12s} recall@1={stats['recall@1']:.0%} "
                f"recall@3={stats['recall@3']:.0%}"
            )

    for name, flags in ({} if configurations else CONFIGURATIONS).items():
        ranks: list[int | None] = []
        for item in GROUND_TRUTH:
            if name == "baseline":
                hits = baseline_retriever.search(item["query"], k=TOP_K)
            else:
                hits = improved_retriever.search(item["query"], k=TOP_K, **flags)
            ranks.append(gold_rank(hits, item))
        configurations[name] = {
            "recall@1": recall_at(ranks, 1),
            "recall@3": recall_at(ranks, 3),
            "ranks": ranks,
        }
        print(
            f"[ok] {name:12s} recall@1={configurations[name]['recall@1']:.0%} "
            f"recall@3={configurations[name]['recall@3']:.0%}"
        )

    rows: list[dict] = []
    for item in GROUND_TRUTH:
        baseline_hits = baseline_retriever.search(item["query"], k=TOP_K)
        improved_hits = improved_retriever.search(item["query"], k=TOP_K)
        filters = detect_filters(item["query"])

        rows.append(
            {
                "query": item["query"],
                "article": item["article"],
                "document_id": item["document_id"],
                "filters": filters.describe(),
                "baseline_top1": baseline_hits[0].chunk_id,
                "baseline_section": baseline_hits[0].metadata["section"],
                "baseline_gold_rank": gold_rank(baseline_hits, item),
                "improved_top1": improved_hits[0].chunk_id,
                "improved_section": improved_hits[0].metadata["section"],
                "improved_gold_rank": gold_rank(improved_hits, item),
                "explanation": explain(
                    baseline_hits[0],
                    improved_hits[0],
                    filters,
                    item,
                    gold_rank(baseline_hits, item),
                    gold_rank(improved_hits, item),
                ),
            }
        )

    demos: list[dict] = []
    for demo in FILTER_DEMOS:
        filters = detect_filters(demo["query"])
        candidates = improved_retriever.base.search(demo["query"], k=20)
        after = [hit for hit in candidates if filters.matches(hit.metadata)]
        without = improved_retriever.search(
            demo["query"], k=1, use_filters=False, use_dedup=False, use_rerank=False
        )
        with_filter = improved_retriever.search(
            demo["query"], k=1, use_filters=True, use_dedup=False, use_rerank=False
        )
        demos.append(
            {
                "query": demo["query"],
                "expect": demo["expect"],
                "filters": filters.describe(),
                "candidates_before": len(candidates),
                "candidates_after": len(after),
                "top1_without": without[0].chunk_id if without else "—",
                "top1_with": with_filter[0].chunk_id if with_filter else "—",
            }
        )
        print(
            f"[filter] {filters.describe():44s} {len(candidates)} -> {len(after)}"
        )

    payload = {
        "generated_at": date.today().isoformat(),
        "queries_tested": len(GROUND_TRUTH),
        "top_k": TOP_K,
        "configurations": configurations,
        "filter_demos": demos,
        "queries": rows,
    }

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(render(payload), encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
