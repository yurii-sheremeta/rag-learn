"""Measure what the structured refusal signal changed.

Compares two stored eval runs of the same ten cases:

    outputs/eval_runs_before.json   phrase matching decided whether the
                                    assistant had declined
    outputs/eval_runs.json          the assistant calls a tool to say so

Both files are complete traces, so the comparison needs no API calls.

The metric that matters is detector accuracy against what actually happened:
cases 4 and 8 are the two questions whose sources genuinely do not answer
them, so a refusal there is correct and anywhere else is not.

Usage:  python scripts/compare_refusal.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE_PATH = ROOT / "outputs" / "eval_runs_before.json"
AFTER_PATH = ROOT / "outputs" / "eval_runs.json"
REPORT_PATH = ROOT / "outputs" / "refusal_comparison.md"

# Ground truth: the sources do not answer these two questions.
SHOULD_REFUSE = {4, 8}


def confusion(flags: dict[int, bool]) -> dict[str, int]:
    result = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for case_id, flagged in flags.items():
        should = case_id in SHOULD_REFUSE
        if should and flagged:
            result["tp"] += 1
        elif should and not flagged:
            result["fn"] += 1
        elif not should and flagged:
            result["fp"] += 1
        else:
            result["tn"] += 1
    return result


def main() -> None:
    if not BEFORE_PATH.exists() or not AFTER_PATH.exists():
        raise SystemExit("both eval_runs_before.json and eval_runs.json are needed")

    before = json.loads(BEFORE_PATH.read_text(encoding="utf-8"))["runs"]
    after = json.loads(AFTER_PATH.read_text(encoding="utf-8"))["runs"]

    # Before: the original detector matched exactly one phrase.
    original = {
        int(k): "недостатньо інформації" in r["answer"].lower()
        for k, r in before.items()
    }
    # After: the flag comes off the tool call.
    structured = {int(k): r["refused"] for k, r in after.items()}
    # The patched regex, for reference — still lexical, still guessing.
    lexical_after = {int(k): r.get("refused_lexical", False) for k, r in after.items()}

    original_stats = confusion(original)
    structured_stats = confusion(structured)

    rows = []
    for case_id in sorted(structured):
        should = case_id in SHOULD_REFUSE
        rows.append(
            {
                "id": case_id,
                "question": after[str(case_id)]["question"],
                "should_refuse": should,
                "before": original[case_id],
                "after": structured[case_id],
                "after_lexical": lexical_after[case_id],
                "tools": after[str(case_id)]["tools_used"],
                "weak": after[str(case_id)].get("weak_context_seen", False),
            }
        )

    def verdict(flagged: bool, should: bool) -> str:
        if should and flagged:
            return "коректно"
        if should and not flagged:
            return "**пропуск**"
        if not should and flagged:
            return "**хибна тривога**"
        return "коректно"

    lines = [
        "# Before / after: сигнал відмови",
        "",
        f"Згенеровано: {date.today().isoformat()} · без жодного виклику API — "
        "порівнюються два збережені прогони",
        "",
        "Еталон: джерела не відповідають на питання 4 (ставки податків) і 8 "
        "(відпустка в Польщі). Відмова там коректна, будь-де інде — ні.",
        "",
        "| id | Питання | Має відмовитись | Було | Стало |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['question'][:42]} | "
            f"{'так' if row['should_refuse'] else 'ні'} | "
            f"{verdict(row['before'], row['should_refuse'])} | "
            f"{verdict(row['after'], row['should_refuse'])} |"
        )

    lines += [
        "",
        "## Точність детектора",
        "",
        "| | Було (одна фраза) | Стало (виклик інструмента) |",
        "|---|---|---|",
        f"| Правильно виявлені відмови | {original_stats['tp']}/2 | "
        f"**{structured_stats['tp']}/2** |",
        f"| Пропущені відмови | **{original_stats['fn']}** | "
        f"{structured_stats['fn']} |",
        f"| Хибні тривоги | {original_stats['fp']} | {structured_stats['fp']} |",
        f"| Правильно пропущені | {original_stats['tn']}/8 | "
        f"{structured_stats['tn']}/8 |",
        "",
        "Пропущена відмова коштує дорожче, ніж виглядає: механічне правило "
        "позначало такі кейси як `hallucination_risk`, тобто моніторинг "
        "показував галюцинацію там, де система поводилась зразково.",
        "",
        "## Чи потрібен ще лексичний детектор",
        "",
        "| id | Структурний сигнал | Лексичний (виправлений) | Збіг |",
        "|---|---|---|---|",
    ]
    disagreements = 0
    for row in rows:
        agree = row["after"] == row["after_lexical"]
        disagreements += 0 if agree else 1
        lines.append(
            f"| {row['id']} | {'так' if row['after'] else 'ні'} | "
            f"{'так' if row['after_lexical'] else 'ні'} | "
            f"{'так' if agree else '**ні**'} |"
        )

    lines += [
        "",
        f"Розбіжностей: **{disagreements} із {len(rows)}**.",
        "",
        "Навіть виправлений регулярний вираз лишається здогадкою про намір за "
        "текстом. Структурний сигнал не здогадується: або інструмент "
        "викликано, або ні.",
        "",
        "## Що ще змінилось у слідах",
        "",
        "| id | Інструменти | Слабкий контекст |",
        "|---|---|---|",
    ]
    for row in rows:
        tools = ", ".join(f"`{t}`" for t in dict.fromkeys(row["tools"])) or "—"
        lines.append(f"| {row['id']} | {tools} | {'так' if row['weak'] else 'ні'} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("detector accuracy (2 cases should refuse):")
    print(f"  before: tp={original_stats['tp']} fn={original_stats['fn']} "
          f"fp={original_stats['fp']}")
    print(f"  after:  tp={structured_stats['tp']} fn={structured_stats['fn']} "
          f"fp={structured_stats['fp']}")
    print(f"structured vs lexical disagreements: {disagreements}/{len(rows)}")
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
