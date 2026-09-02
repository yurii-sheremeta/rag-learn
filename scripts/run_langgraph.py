"""Trace the LangGraph port and prove it behaves like the hand-written flow.

"I ported the workflow" is a claim. Running both implementations over the same
questions and diffing route, slots, tool calls and final answer is a check —
and it is cheap, because neither implementation calls a model.

Outputs:
    outputs/langgraph_examples.md   traces plus the generated graph diagram
    outputs/parity_check.md         field-by-field diff, custom vs LangGraph

Usage:  python scripts/run_langgraph.py
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

from agent_flow import run_flow
from langgraph_flow import APP, run

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "outputs" / "langgraph_examples.md"
PARITY_PATH = ROOT / "outputs" / "parity_check.md"

QUESTIONS: list[dict[str, str]] = [
    {
        "question": "Яка зараз мінімальна заробітна плата?",
        "probes": "проста гілка: один інструмент, три вузли",
    },
    {
        "question": "Скільки днів щорічної відпустки належить за законом?",
        "probes": "гілка бази знань — тут інструмент не потрібен",
    },
    {
        "question": "Подай заявку на відпустку для emp_001 з 01.10.2026 на 10 днів",
        "probes": "найдовший шлях: шість вузлів, два інструменти, стан між ними",
    },
    {
        "question": "Хочу подати заявку на відпустку",
        "probes": "другий conditional edge: слотів бракує → ask_user",
    },
    {
        "question": "Розкажи щось цікаве",
        "probes": "перший conditional edge веде повз усі гілки",
    },
]


def render_examples(results: list[dict], diagram: str) -> str:
    lines = [
        "# Трасування LangGraph workflow",
        "",
        "Той самий workflow, що в ДЗ №6, перенесений на LangGraph. Правила "
        "маршрутизації, витяг слотів, інструменти й шаблони відповідей "
        "імпортовані з `agent_flow.py` — змінилась **лише оркестрація**.",
        "",
        f"Згенеровано: {date.today().isoformat()} · "
        "відтворюється командою `python scripts/run_langgraph.py`",
        "",
        "## Граф",
        "",
        "Діаграма згенерована самим фреймворком "
        "(`APP.get_graph().draw_mermaid()`) — вона не намальована вручну, "
        "тому не може розійтися з кодом.",
        "",
        "```mermaid",
        diagram.rstrip(),
        "```",
        "",
        "---",
        "",
    ]

    for number, item in enumerate(results, start=1):
        state = item["state"]
        lines.append(f"## {number}. {state['user_question']}")
        lines.append("")
        lines.append(f"*Перевіряє: {item['probes']}*")
        lines.append("")
        lines.append("```")
        lines.append(f"Input: {state['user_question']}")
        lines.append(f"Route: {state['selected_route']}")
        lines.append(f"  reason: {state.get('route_reason', '')}")
        lines.append(f"Nodes executed: {' -> '.join(state['nodes_executed'])}")
        lines.append("")

        for observation in state.get("observations", []):
            lines.append(f"Tool: {observation['tool']}")
            lines.append(
                f"  Input: {json.dumps(observation['input'], ensure_ascii=False)}"
            )
            result = json.dumps(observation["result"], ensure_ascii=False)
            for offset, line in enumerate(textwrap.wrap(result, width=84)[:4]):
                lines.append(f"{'  Result:' if offset == 0 else '         '} {line}")
            lines.append("")

        lines.append("Final state:")
        summary = {
            "user_question": state["user_question"],
            "selected_route": state["selected_route"],
            "slots": state.get("slots", {}),
            "nodes_executed": state["nodes_executed"],
            "needs_user_input": state.get("needs_user_input"),
        }
        for line in json.dumps(summary, ensure_ascii=False, indent=2).splitlines():
            lines.append(f"  {line}")
        lines.append("")
        lines.append("Final answer:")
        for paragraph in (state.get("final_answer") or "").split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=86) or [""])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def compare(question: str) -> dict:
    """Run both implementations and diff what a caller would actually observe."""
    custom = run_flow(question)
    graph = run(question)

    custom_tools = [step.tool for step in custom.steps if step.tool]
    graph_tools = [obs["tool"] for obs in graph.get("observations", [])]

    fields = {
        "route": (custom.selected_route, graph["selected_route"]),
        "slots": (custom.slots, graph.get("slots", {})),
        "tools": (custom_tools, graph_tools),
        "needs_user_input": (custom.needs_user_input, graph.get("needs_user_input")),
        "final_answer": (custom.final_answer, graph.get("final_answer")),
    }
    differences = [name for name, (a, b) in fields.items() if a != b]

    return {
        "question": question,
        "route": custom.selected_route,
        "custom_tools": custom_tools,
        "graph_tools": graph_tools,
        "differences": differences,
        "identical": not differences,
    }


def render_parity(rows: list[dict]) -> str:
    identical = sum(1 for row in rows if row["identical"])
    lines = [
        "# Перевірка паритету: custom flow проти LangGraph",
        "",
        "Обидві реалізації прогнані на тих самих питаннях. Порівнюються поля, "
        "які реально бачить користувач: обраний маршрут, витягнуті слоти, "
        "перелік викликаних інструментів, запит на уточнення й фінальна "
        "відповідь.",
        "",
        f"Згенеровано: {date.today().isoformat()}",
        "",
        f"**Ідентичний результат: {identical} із {len(rows)}.**",
        "",
        "| Питання | Маршрут | Інструменти (custom) | Інструменти (граф) | Розбіжності |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        custom = ", ".join(f"`{t}`" for t in row["custom_tools"]) or "—"
        graph = ", ".join(f"`{t}`" for t in row["graph_tools"]) or "—"
        diff = ", ".join(row["differences"]) if row["differences"] else "немає"
        lines.append(
            f"| {row['question'][:44]} | `{row['route']}` | {custom} | {graph} | {diff} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Чому це варто перевіряти, а не декларувати",
        "",
        "Перенесення workflow на фреймворк легко зробити «майже правильно»: "
        "загубити гілку, переплутати порядок кроків, зібрати відповідь з "
        "іншого поля стану. Жодна з цих помилок не викликає винятку — вона "
        "просто тихо змінює поведінку.",
        "",
        "Оскільки логіка імпортована з `agent_flow.py`, а не скопійована, "
        "розбіжність у цій таблиці означала б помилку саме в **графі**: "
        "неправильне ребро, пропущений вузол або не те поле стану. Це "
        "перетворює порівняння на регресійний тест для порту.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    diagram = APP.get_graph().draw_mermaid()

    results = []
    for item in QUESTIONS:
        state = run(item["question"])
        results.append({**item, "state": state})
        print(
            f"[ok] {item['question'][:46]:48s} "
            f"{state['selected_route']:22s} "
            f"{len(state['nodes_executed'])} вузлів"
        )

    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(render_examples(results, diagram), encoding="utf-8")
    print(f"\nWrote {EXAMPLES_PATH.relative_to(ROOT)}")

    rows = [compare(item["question"]) for item in QUESTIONS]
    PARITY_PATH.write_text(render_parity(rows), encoding="utf-8")
    identical = sum(1 for row in rows if row["identical"])
    print(f"Parity: {identical}/{len(rows)} identical")
    for row in rows:
        if row["differences"]:
            print(f"  DIFF {row['question'][:40]}: {', '.join(row['differences'])}")
    print(f"Wrote {PARITY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
