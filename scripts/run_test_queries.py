"""Run the fixed test-query set and render outputs/retrieval_examples.md.

The queries are phrased the way an employee would actually ask them — not as
quotations from the acts. Quoting the statute back at the index would make
every top-1 a trivial hit and leave nothing to analyse.

`comment` holds the manual assessment of the observed results (relevant /
partially relevant / not relevant plus the reason). It is stored next to the
query so that re-running the script reproduces the report without losing the
analysis.

Usage:  python scripts/run_test_queries.py
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

from retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "retrieval_examples.md"
TOP_K = 3

TEST_QUERIES: list[dict[str, str]] = [
    {
        "query": "Скільки днів щорічної відпустки мені належить?",
        "expected": "Стаття 6 Закону «Про відпустки» / стаття 75 КЗпП — 24 календарних дні",
        "comment": (
            "relevant — Top-1 містить пряму відповідь (24 календарних дні). Але всі три слоти зайняла та сама стаття 6, розрізана на послідовні чанки; еквівалентна стаття 75 КЗпП не потрапила у видачу. Відповідь є, різноманітності немає."
        ),
    },
    {
        "query": "Чи можуть звільнити працівника під час лікарняного?",
        "expected": "Стаття 40 КЗпП — заборона звільнення в період тимчасової непрацездатності",
        "comment": (
            "partially relevant — Top-1 (стаття 5 закону про воєнний стан) дає чинний виняток і фактично найактуальнішу відповідь на сьогодні. Top-2 — потрібна стаття 40 КЗпП, але саме той її чанк, що містить перелік підстав, а не частину із забороною звільнення під час непрацездатності. Висновок користувач зробить правильний, базової норми не побачить."
        ),
    },
    {
        "query": "Коли роботодавець має виплатити розрахунок при звільненні?",
        "expected": "Стаття 116 КЗпП — у день звільнення",
        "comment": (
            "relevant — найкращий результат набору. Top-1 (стаття 47, обов'язок провести розрахунок), Top-2 (стаття 116, строк — день звільнення), Top-3 (стаття 117, відповідальність за затримку) разом складають повну відповідь із трьох взаємодоповнюваних норм."
        ),
    },
    {
        "query": "Чи можна встановити випробувальний термін під час воєнного стану?",
        "expected": "Стаття 2 Закону про трудові відносини в умовах воєнного стану",
        "comment": (
            "relevant — Top-1 і Top-2 точні (стаття 2, обидва чанки однієї статті). Top-3 (стаття 16, державний нагляд) — шум: підтягнувся за спільною фразою «у період дії воєнного стану», змістовного стосунку до запиту не має."
        ),
    },
    {
        "query": "Хто розслідує нещасний випадок на виробництві?",
        "expected": "Закон «Про охорону праці» — розслідування та облік нещасних випадків",
        "comment": (
            "relevant — Top-1 точна стаття 22. Top-2 — стаття 171 КЗпП, тобто дубль тієї самої норми з іншого акта. Наочна ілюстрація проблеми дублювання, передбаченої у висновках ДЗ №1."
        ),
    },
    {
        "query": "Що таке мінімальна заробітна плата?",
        "expected": "Стаття 3 Закону «Про оплату праці»",
        "comment": (
            "relevant, але половина видачі — дубль. Top-1 (стаття 3 закону) і Top-2 (стаття 95 КЗпП) починаються буквально однаковим реченням, різниця скорів 0.001. Один зі слотів top-3 витрачено намарно."
        ),
    },
    {
        "query": "Скільки годин на тиждень я маю працювати?",
        "expected": "Стаття 50 КЗпП — не більше 40 годин на тиждень",
        "comment": (
            "partially relevant — правильна норма (стаття 50, «не може перевищувати 40 годин на тиждень») лише на 2-му місці. Top-1 — стаття 52 про п'ятиденний і шестиденний тиждень: її заголовок лексично ближчий до запиту («робочий тиждень», «тривалість»), хоча прямої відповіді вона не містить."
        ),
    },
    {
        "query": "Мене змушують вийти на роботу у вихідний, чи це законно?",
        "expected": "Стаття 71 КЗпП — заборона залучення до роботи у вихідні, крім винятків",
        "comment": (
            "partially relevant — стаття вгадана точно (стаття 71), але Top-1 — середина переліку винятків, що починається з «вантажно-розвантажувальних робіт». Саме речення із забороною лишилось у сусідньому чанку. З Top-1 користувач відповіді «чи це законно» не отримає."
        ),
    },
    {
        "query": "Чи можу я взяти відпустку за свій рахунок на два тижні?",
        "expected": "Стаття 26 Закону «Про відпустки» — відпустка без збереження зарплати",
        "comment": (
            "partially relevant, найгірший і найкорисніший кейс. Статті 25 і 26 Закону «Про відпустки» не потрапили навіть у топ-10: стаття 25 розрізана на 10 чанків-переліків категорій, і визначальне речення тоне серед «пенсіонерам… ветеранам праці… сумісникам». Врятувала видачу стаття 84 КЗпП на 2-му місці — компактне формулювання тієї ж норми. Top-1 (грошова компенсація за невикористані відпустки) і Top-3 (відрахування із зарплати) хибні."
        ),
    },
    {
        "query": "Що робити, якщо роботодавець не виконує колективний договір?",
        "expected": "Закон «Про колективні договори і угоди» — відповідальність за невиконання",
        "comment": (
            "partially relevant — потрібна стаття 18 («Відповідальність за порушення і невиконання») на 2-му місці. Top-1 — стаття 4 про порядок ведення переговорів: закон вгадано, статтю ні."
        ),
    },
]


def render(results: list[dict]) -> str:
    config = json.loads((ROOT / "index" / "config.json").read_text(encoding="utf-8"))

    lines = [
        "# Приклади semantic retrieval",
        "",
        f"Модель: `{config['model']}` · {config['vectors']:,} векторів × "
        f"{config['dimension']} вимірів · {config['index_type']} · "
        f"{config['similarity']}",
        "",
        f"Згенеровано: {date.today().isoformat()} · "
        f"відтворюється командою `python scripts/run_test_queries.py`",
        "",
        "Score — косинусна подібність у діапазоні `[-1, 1]`; більше = ближче.",
        "",
        "---",
        "",
    ]

    for number, item in enumerate(results, start=1):
        lines.append(f"## {number}. {item['query']}")
        lines.append("")
        lines.append("```")
        lines.append(f"Query: {item['query']}")
        lines.append("")
        for hit in item["hits"]:
            meta = hit["metadata"]
            lines.append(
                f"Top-{hit['rank']}: {hit['chunk_id']} | score: {hit['score']:.4f}"
            )
            lines.append(f"  Section: {meta.get('section')}")
            lines.append(f"  Source:  {meta.get('source_file')} ({meta.get('document_id')})")
            for offset, line in enumerate(
                textwrap.wrap(hit["preview"], width=86)
            ):
                lines.append(f"  {'Text:   ' if offset == 0 else '        '} {line}")
            lines.append("")
        for offset, line in enumerate(
            textwrap.wrap(item["comment"] or "TODO", width=86)
        ):
            lines.append(f"{'Comment:' if offset == 0 else '        '} {line}")
        lines.append("```")
        lines.append("")
        lines.append(f"**Очікувана норма:** {item['expected']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    retriever = Retriever()
    print(f"Model: {retriever.config['model']}")

    results: list[dict] = []
    for item in TEST_QUERIES:
        hits = retriever.search(item["query"], k=TOP_K)
        results.append(
            {
                "query": item["query"],
                "expected": item["expected"],
                "comment": item["comment"],
                "hits": [hit.to_dict() for hit in hits],
            }
        )
        top = hits[0]
        print(
            f"[ok] {item['query'][:52]:52s} -> {top.chunk_id} ({top.score:.3f})"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(results), encoding="utf-8")

    (ROOT / "outputs" / "retrieval_examples.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
