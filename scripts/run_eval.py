"""Run the eval set through the assistant and produce the observability layer.

Evaluates the real chatbot — the model-driven agent from HW5, which picks
between the knowledge base and the three tools. Evaluating the deterministic
flow instead would be easier and would measure the wrong thing.

Outputs:
    outputs/eval_runs.json      raw traces, written per case so a crash keeps
                                what was already paid for
    outputs/eval_results.csv    the eval table, one row per case
    outputs/eval_results.md     same table, readable
    outputs/eval_summary.md     observability metrics
    outputs/quality_report.md   the written analysis

Which columns are measured and which are judged is kept explicit. Route,
tools, citations, refusal and latency come from the trace. task_success,
groundedness and answer_quality are read off the answers by hand and stored
in eval_set.LABELS — a script that scored its own output would only be
measuring its own heuristics.

Usage:
    python scripts/run_eval.py            # run missing cases, render reports
    python scripts/run_eval.py --render   # re-render from stored runs, no calls
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path

from eval_set import EVAL_CASES, LABELS

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "outputs" / "eval_runs.json"
CSV_PATH = ROOT / "outputs" / "eval_results.csv"
TABLE_PATH = ROOT / "outputs" / "eval_results.md"
SUMMARY_PATH = ROOT / "outputs" / "eval_summary.md"

TOOL_TO_ROUTE = {
    "get_statutory_amount": "statutory_amount",
    "calculate_vacation_entitlement": "personal_calculation",
    "submit_leave_request": "leave_request",
    "search_labour_law": "legal_norm",
}

CHUNK_RE = re.compile(r"[a-z_]+_chunk_\d{4}")

# Detecting "the system declined" turned out to be the hardest metric here.
# The first version matched one phrase — the fixed sentence the HW4 prompt
# mandates. The HW5 agent has no such rule, so it declines in its own words
# and the detector scored two correct refusals as hallucination risk. Two
# false alarms out of two: monitoring that would have sent us to fix a system
# that was working.
#
# The patterns below cover the phrasings actually observed. This is still a
# heuristic, and the real fix is structural rather than lexical — see the
# quality report.
REFUSAL_PATTERNS = re.compile(
    r"недостатньо інформації"
    r"|(?:у )?мо(?:їй|їй) баз[іи] немає"
    r"|інформації в мене немає"
    r"|немає (?:в|у) (?:моїй )?баз[іи]"
    r"|не можу (?:назвати|відповісти)"
    r"|(?:назвати|відповісти)[^.]{0,40}не можу",
    re.IGNORECASE,
)


def detect_refusal_lexically(answer: str) -> bool:
    """The old detector. Kept only to measure how far off it was."""
    return bool(REFUSAL_PATTERNS.search(answer))


def classify_error(case: dict, record: dict) -> str:
    """Mechanical error signals. Anything subtler is left to the labels."""
    errors: list[str] = []

    if not [t for t in record["tools_used"] if t != "report_insufficient_context"]:
        errors.append("no_tool_call")
    elif case["expected_route"] not in record["routes"]:
        errors.append("wrong_route")

    if case["kind"] in {"must_refuse", "general_knowledge_trap"} and not record["refused"]:
        errors.append("hallucination_risk")

    if (
        case["kind"] not in {"must_refuse", "general_knowledge_trap"}
        and not record["refused"]
        and "legal_norm" in record["routes"]
        and not record["cited_chunks"]
    ):
        errors.append("missing_citation")

    return ", ".join(errors) if errors else "none"


def run_case(case: dict) -> dict:
    from rag_agent import ask

    started = time.perf_counter()
    result = ask(case["question"])
    latency_ms = int((time.perf_counter() - started) * 1000)

    tools_used = [call.name for call in result.calls]
    routes = list(dict.fromkeys(TOOL_TO_ROUTE.get(t, t) for t in tools_used))

    retrieved: list[str] = []
    for call in result.calls:
        if call.name == "search_labour_law" and isinstance(call.result, dict):
            retrieved.extend(c["chunk_id"] for c in call.result.get("chunks", []))

    return {
        "id": case["id"],
        "question": case["question"],
        "kind": case["kind"],
        "expected_behavior": case["expected_behavior"],
        "expected_route": case["expected_route"],
        "answer": result.text,
        "retrieved_chunks": retrieved,
        "cited_chunks": sorted(set(CHUNK_RE.findall(result.text))),
        "routes": routes,
        "tools_used": tools_used,
        "refused": result.refused,
        "refused_lexical": detect_refusal_lexically(result.text),
        "refusal_details": result.refusal_details,
        "weak_context_seen": result.weak_context_seen,
        "iterations": result.iterations,
        "latency_ms": latency_ms,
        "cost_usd": round(result.cost_usd, 5),
    }


def route_or_mode(record: dict) -> str:
    if record["refused"]:
        return "fallback"
    if not record["routes"]:
        return "clarification"
    if record["routes"] == ["legal_norm"]:
        return "RAG"
    if "legal_norm" in record["routes"]:
        return "RAG + tool"
    return "tool"


def rows_for_report(store: dict) -> list[dict]:
    rows = []
    for case in EVAL_CASES:
        record = store["runs"].get(str(case["id"]))
        if record is None:
            continue
        label = LABELS.get(case["id"], {})
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "answer": record["answer"],
                "retrieved_chunks": ", ".join(record["retrieved_chunks"]) or "—",
                "route_or_mode": route_or_mode(record),
                "tools_used": ", ".join(dict.fromkeys(record["tools_used"])) or "—",
                "task_success": label.get("task_success", "не оцінено"),
                "groundedness": label.get("groundedness", "не оцінено"),
                "answer_quality": label.get("answer_quality", "не оцінено"),
                "latency_ms": record["latency_ms"],
                "errors": classify_error(case, record),
                "notes": label.get("notes", ""),
            }
        )
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        "# Eval table",
        "",
        "Оцінювався реальний чат-бот — LLM-агент із ДЗ №5, який сам обирає між "
        "базою знань і трьома інструментами.",
        "",
        f"Згенеровано: {date.today().isoformat()} · "
        "відтворюється командою `python scripts/run_eval.py --render`",
        "",
        "| id | question | expected_behavior | route_or_mode | tools_used | "
        "task_success | groundedness | answer_quality | latency_ms | errors | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['question'][:46]} | {row['expected_behavior'][:44]} | "
            f"{row['route_or_mode']} | {row['tools_used']} | **{row['task_success']}** | "
            f"{row['groundedness']} | {row['answer_quality']} | {row['latency_ms']} | "
            f"`{row['errors']}` | {row['notes']} |"
        )

    lines += ["", "---", "", "## Відповіді та джерела", ""]
    for row in rows:
        lines.append(f"### {row['id']}. {row['question']}")
        lines.append("")
        lines.append(f"- **Очікувалось:** {row['expected_behavior']}")
        lines.append(f"- **Режим:** {row['route_or_mode']} · інструменти: {row['tools_used']}")
        lines.append(f"- **Знайдені чанки:** {row['retrieved_chunks']}")
        lines.append(f"- **Затримка:** {row['latency_ms']} мс · помилки: `{row['errors']}`")
        lines.append("")
        lines.append("```")
        lines.append(row["answer"][:900] + ("…" if len(row["answer"]) > 900 else ""))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_summary(rows: list[dict], store: dict) -> str:
    total = len(rows)
    success = Counter(row["task_success"] for row in rows)
    grounded = Counter(row["groundedness"] for row in rows)
    quality = Counter(row["answer_quality"] for row in rows)
    latencies = [row["latency_ms"] for row in rows]

    error_counter: Counter[str] = Counter()
    for row in rows:
        for error in row["errors"].split(", "):
            error_counter[error] += 1

    modes = Counter(row["route_or_mode"] for row in rows)
    cost = sum(r["cost_usd"] for r in store["runs"].values())

    def pct(value: int) -> str:
        return f"{value}/{total} = {value / total:.0%}"

    lines = [
        "# Observability metrics",
        "",
        f"Згенеровано: {date.today().isoformat()}",
        "",
        "```",
        f"Total cases:            {total}",
        "",
        f"Success rate:           {pct(success['yes'])}",
        f"Partial success:        {pct(success['partial'])}",
        f"Failure rate:           {pct(success['no'])}",
        "",
        f"Groundedness good:      {pct(grounded['good'])}",
        f"Groundedness partial:   {pct(grounded['partial'])}",
        f"Groundedness bad:       {pct(grounded['bad'])}",
        f"Groundedness n/a:       {pct(grounded['not_applicable'])}",
        "",
        f"Answer quality good:    {pct(quality['good'])}",
        f"Answer quality partial: {pct(quality['partial'])}",
        f"Answer quality bad:     {pct(quality['bad'])}",
        "",
        f"Average latency:        {statistics.mean(latencies):,.0f} ms",
        f"Median latency:         {statistics.median(latencies):,.0f} ms",
        f"Min latency:            {min(latencies):,} ms",
        f"Max latency:            {max(latencies):,} ms",
        "",
        "Error types:",
    ]
    for name, count in error_counter.most_common():
        lines.append(f"  {name + ':':<22} {count}")

    lines += ["", "Routes taken:"]
    for name, count in modes.most_common():
        lines.append(f"  {name + ':':<22} {count}")

    lines += [
        "",
        f"Total cost of eval run: ${cost:.4f}",
        "```",
        "",
        "---",
        "",
        "## Що вимірюється автоматично, а що оцінено вручну",
        "",
        "Розділення навмисне, бо змішувати їх — значить видавати власні "
        "евристики за об'єктивні метрики.",
        "",
        "| Колонка | Джерело |",
        "|---|---|",
        "| `route_or_mode`, `tools_used` | зі сліду виконання |",
        "| `retrieved_chunks` | з результату `search_labour_law` |",
        "| `latency_ms` | вимірювання навколо виклику |",
        "| `errors` | механічні правила: не той маршрут, відсутня цитата, відсутня відмова там, де вона потрібна |",
        "| `task_success`, `groundedness`, `answer_quality` | **прочитано вручну** і збережено в `eval_set.LABELS` |",
        "",
        "Мітки лежать у коді як дані, а не в голові автора: звіт "
        "перегенеровується без повторного платного прогону, а самі оцінки "
        "можна перевірити рядок за рядком.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict]) -> None:
    columns = [
        "id", "question", "expected_behavior", "answer", "retrieved_chunks",
        "route_or_mode", "tools_used", "task_success", "groundedness",
        "answer_quality", "latency_ms", "errors", "notes",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="no API calls")
    args = parser.parse_args()

    store = (
        json.loads(RUNS_PATH.read_text(encoding="utf-8"))
        if RUNS_PATH.exists()
        else {"generated_at": date.today().isoformat(), "runs": {}}
    )

    if not args.render:
        for case in EVAL_CASES:
            if str(case["id"]) in store["runs"]:
                continue
            record = run_case(case)
            store["runs"][str(case["id"])] = record
            RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RUNS_PATH.write_text(
                json.dumps(store, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[ok] {case['id']:2d} {case['kind']:24s} "
                f"{route_or_mode(record):12s} {record['latency_ms']:6d} ms "
                f"${record['cost_usd']:.4f}"
            )

    rows = rows_for_report(store)
    TABLE_PATH.write_text(render_table(rows), encoding="utf-8")
    SUMMARY_PATH.write_text(render_summary(rows, store), encoding="utf-8")
    write_csv(rows)

    unlabelled = [row["id"] for row in rows if row["task_success"] == "не оцінено"]
    print(f"\nWrote {TABLE_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    if unlabelled:
        print(f"\nUnlabelled cases: {unlabelled} — fill eval_set.LABELS")


if __name__ == "__main__":
    main()
