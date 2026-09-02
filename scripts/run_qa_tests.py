"""Run the test questions through one or more prompt versions and report.

Every answer is written to outputs/rag_answers.json as soon as it arrives:
each call costs money, and a crash halfway through should not throw away the
answers already paid for. Rerunning skips whatever is already stored.

Outputs:
    outputs/rag_answers.json          every answer, machine readable
    outputs/rag_answers_examples.md   the graded report for the final prompt
    outputs/prompt_improvements.md    v1 vs v2 vs v3, before and after

Usage:
    python scripts/run_qa_tests.py
    python scripts/run_qa_tests.py --versions v3
    python scripts/run_qa_tests.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import date
from pathlib import Path

from prompts import CHANGELOG
from qa_questions import TEST_QUESTIONS
from rag_answer import answer, load_cached_chunks

ROOT = Path(__file__).resolve().parents[1]
ANSWERS_PATH = ROOT / "outputs" / "rag_answers.json"
EXAMPLES_PATH = ROOT / "outputs" / "rag_answers_examples.md"
IMPROVEMENTS_PATH = ROOT / "outputs" / "prompt_improvements.md"

FINAL_VERSION = "v3"

KIND_LABELS = {
    "grounded": "відповідь є в корпусі",
    "reformulated": "переформульоване питання",
    "multi_source": "потрібно кілька статей",
    "weak_context": "слабкий retrieval",
    "out_of_scope": "немає в корпусі",
    "general_trap": "пастка на загальні знання",
}

def should_refuse(item: dict) -> bool:
    """A refusal is correct exactly when the context lacks the answer.

    Judged by the verified `context_sufficient` flag, not by question type:
    q03 is a fair reformulation whose retrieval happened to miss the norm, so
    scoring it by type would have counted a correct refusal as a failure.
    """
    return not item["context_sufficient"]


def load_answers() -> dict:
    if ANSWERS_PATH.exists():
        return json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    return {"generated_at": date.today().isoformat(), "answers": {}}


def save_answers(store: dict) -> None:
    ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verdict(item: dict, record: dict) -> str:
    """A factual reading of what the model did, judged against the question type."""
    refused = record["refused"]
    cited = record["cited_chunk_ids"]
    must_refuse = should_refuse(item)

    if must_refuse and refused:
        return "коректно — контекст не містить відповіді, модель це визнала"
    if must_refuse and not refused:
        return (
            "**помилка** — модель відповіла, хоча відповіді в контексті немає: "
            "ознака використання знань поза контекстом"
        )
    if refused:
        return "**помилка** — відповідь була в контексті, але модель відмовилась"
    if not cited:
        return "відповідь по суті, але **без цитати** chunk_id"
    return f"коректно — відповідь із контексту, цитовано {len(cited)} чанк(ів)"


def render_examples(store: dict, version: str) -> str:
    lines = [
        "# Приклади grounded QA",
        "",
        f"Модель: `claude-opus-5` · промпт `{version}` · "
        f"контекст: top-3 чанки з покращеного retrieval (ДЗ №3)",
        "",
        f"Згенеровано: {store['generated_at']} · "
        "відтворюється командою `python scripts/run_qa_tests.py`",
        "",
        "---",
        "",
    ]

    for number, item in enumerate(TEST_QUESTIONS, start=1):
        key = f"{item['id']}::{version}"
        record = store["answers"].get(key)
        if record is None:
            continue

        lines.append(f"## {number}. {item['question']}")
        lines.append("")
        lines.append(f"*Тип: {KIND_LABELS.get(item['kind'], item['kind'])}*")
        lines.append("")
        lines.append("```")
        lines.append(f"Question: {item['question']}")
        lines.append("")
        lines.append("Retrieved chunks:")
        for chunk in record["retrieved_chunks"]:
            mark = "*" if chunk["chunk_id"] in record["cited_chunk_ids"] else " "
            lines.append(
                f" {mark} {chunk['chunk_id']} (score {chunk['score']:.4f}) — "
                f"{chunk['section']}"
            )
        lines.append("")
        lines.append("Answer:")
        for paragraph in record["answer"].split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            lines.extend(textwrap.wrap(paragraph, width=88))
        lines.append("")
        lines.append(f"Source: {', '.join(record['sources'])}")
        lines.append("")
        lines.append("Comment: " + record["verdict"])
        lines.append("```")
        lines.append("")
        lines.append(f"**Очікувалось:** {item['expected']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_improvements(store: dict, versions: list[str]) -> str:
    lines = [
        "# Purpose-built prompt improvements",
        "",
        "Промпт `v1` — наївний варіант із формулювання завдання. Його прогнали "
        "по всіх десяти питаннях спеціально, щоб побачити реальні поломки, "
        "а не вигадати їх. `v2` і `v3` виправляють те, що справді зламалось.",
        "",
        "---",
        "",
        "## Версії промпта",
        "",
    ]

    for version in versions:
        lines.append(f"### `{version}`")
        lines.append("")
        lines.append(CHANGELOG.get(version, ""))
        lines.append("")

    lines += ["---", "", "## Поведінка по версіях", ""]

    header = "| # | Питання | Тип | " + " | ".join(f"`{v}`" for v in versions) + " |"
    lines.append(header)
    lines.append("|---|---|---|" + "---|" * len(versions))

    for number, item in enumerate(TEST_QUESTIONS, start=1):
        cells = []
        for version in versions:
            record = store["answers"].get(f"{item['id']}::{version}")
            if record is None:
                cells.append("—")
                continue
            if should_refuse(item):
                cells.append("відмова ✓" if record["refused"] else "**відповів ✗**")
            elif record["refused"]:
                cells.append("**відмова ✗**")
            else:
                cells.append(
                    "з цитатою ✓" if record["cited_chunk_ids"] else "**без цитати ✗**"
                )
        lines.append(
            f"| {number} | {item['question'][:48]} | "
            f"{KIND_LABELS.get(item['kind'], item['kind'])} | "
            + " | ".join(cells)
            + " |"
        )

    lines += ["", "---", "", "## Підсумок по версіях", ""]
    lines.append("| Версія | Відмовилась там, де треба | Цитує джерело | Хибних відповідей |")
    lines.append("|---|---|---|---|")

    for version in versions:
        must_refuse = correct_refusals = cited = wrong = answerable = 0
        for item in TEST_QUESTIONS:
            record = store["answers"].get(f"{item['id']}::{version}")
            if record is None:
                continue
            if should_refuse(item):
                must_refuse += 1
                if record["refused"]:
                    correct_refusals += 1
                else:
                    wrong += 1
            else:
                answerable += 1
                if record["cited_chunk_ids"]:
                    cited += 1
        lines.append(
            f"| `{version}` | {correct_refusals}/{must_refuse} | "
            f"{cited}/{answerable} | {wrong} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", default="v1,v2,v3")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    store = {"generated_at": date.today().isoformat(), "answers": {}}
    if not args.rebuild:
        store = load_answers()
    store["generated_at"] = date.today().isoformat()

    spent = 0.0
    for version in versions:
        for item in TEST_QUESTIONS:
            key = f"{item['id']}::{version}"
            if key in store["answers"]:
                continue

            chunks = load_cached_chunks(item["id"])
            result = answer(item["question"], chunks, version=version)
            record = result.to_dict()
            record["question_id"] = item["id"]
            record["kind"] = item["kind"]
            record["verdict"] = verdict(item, record)
            record["context_sufficient"] = item["context_sufficient"]

            store["answers"][key] = record
            save_answers(store)

            spent += result.cost_usd
            flag = "refused" if result.refused else "answered"
            print(
                f"[{version}] {item['id']:32s} {flag:9s} "
                f"cited={len(result.cited_chunk_ids)} ${result.cost_usd:.4f}"
            )

    EXAMPLES_PATH.write_text(render_examples(store, FINAL_VERSION), encoding="utf-8")
    IMPROVEMENTS_PATH.write_text(render_improvements(store, versions), encoding="utf-8")

    total = sum(r["usage"]["cost_usd"] for r in store["answers"].values())
    print(f"\nSpent this run: ${spent:.4f} | total in store: ${total:.4f}")
    print(f"Wrote {EXAMPLES_PATH.relative_to(ROOT)}")
    print(f"Wrote {IMPROVEMENTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
