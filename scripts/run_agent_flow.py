"""Trace the workflow over the test scenarios and compare it with HW5 routing.

Two outputs, one run, no API calls:

    outputs/agent_flow_examples.md   six traces, route -> steps -> state -> answer
    outputs/routing_comparison.md    where the keyword router and the model
                                     router agree, and where they do not

The comparison is the point. HW5 stored what the model chose for its scenarios
in outputs/tool_runs.json; replaying those same questions through the rule
router turns "deterministic routing is cheaper" from an assertion into a
measurement — including of what the cheap router gets wrong.

Usage:  python scripts/run_agent_flow.py
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

from agent_flow import run_flow

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "outputs" / "agent_flow_examples.md"
COMPARISON_PATH = ROOT / "outputs" / "routing_comparison.md"
HW5_RUNS = ROOT / "outputs" / "tool_runs.json"

SCENARIOS: list[dict] = [
    {
        "question": "Яка зараз мінімальна заробітна плата?",
        "expect": "statutory_amount",
        "probes": "проста маршрутизація на довідник сум",
    },
    {
        "question": "Скільки днів щорічної відпустки належить за законом?",
        "expect": "legal_norm",
        "probes": "норма права — має піти в базу знань, а не в інструмент",
    },
    {
        "question": "Я працюю з 15 березня 2023 року. Скільки днів відпустки я вже накопичив?",
        "expect": "personal_calculation",
        "probes": "витяг дати з тексту словами + розрахунок",
    },
    {
        "question": "Подай заявку на відпустку для emp_001 з 01.10.2026 на 10 днів",
        "expect": "leave_request",
        "probes": "багатокроковий маршрут: слоти → розрахунок → прев'ю → підтвердження",
    },
    {
        "question": "Хочу подати заявку на відпустку",
        "expect": "leave_request → ask_user",
        "probes": "перехід маршруту: слотів бракує, workflow питає користувача",
    },
    {
        "question": "Розкажи щось цікаве",
        "expect": "clarification",
        "probes": "нічого не збіглося — fallback",
    },
]

TOOL_TO_ROUTE = {
    "get_statutory_amount": "statutory_amount",
    "calculate_vacation_entitlement": "personal_calculation",
    "submit_leave_request": "leave_request",
    "search_labour_law": "legal_norm",
}


def render_examples(results: list[dict]) -> str:
    lines = [
        "# Трасування agentic workflow",
        "",
        "Детермінований rule-based роутер — жодного виклику LLM, тому прогін "
        "безкоштовний і повторюється байт у байт.",
        "",
        f"Згенеровано: {date.today().isoformat()} · "
        "відтворюється командою `python scripts/run_agent_flow.py`",
        "",
        "---",
        "",
    ]

    for number, item in enumerate(results, start=1):
        state = item["state"]
        lines.append(f"## {number}. {state['user_goal']}")
        lines.append("")
        lines.append(f"*Перевіряє: {item['probes']}*")
        lines.append("")
        lines.append("```")
        lines.append(f"Question: {state['user_goal']}")
        lines.append(f"Route: {state['selected_route']}")
        lines.append(f"  reason: {state['route_reason']}")
        lines.append(f"Plan: {' -> '.join(state['plan'])}")
        lines.append("")

        for step_number, step in enumerate(state["steps"], start=1):
            flag = "" if step["ok"] else "   [слотів бракує]"
            lines.append(f"Step {step_number}: {step['name']}{flag}")
            if step["tool"]:
                lines.append(f"  Tool called: {step['tool']}")
                lines.append(
                    f"  Input: {json.dumps(step['tool_input'], ensure_ascii=False)}"
                )
            observation = json.dumps(step["observation"], ensure_ascii=False, default=str)
            for offset, line in enumerate(textwrap.wrap(observation, width=84)[:4]):
                lines.append(f"{'  Observation:' if offset == 0 else '              '} {line}")
            lines.append("")

        lines.append("State after run:")
        summary = {
            "user_goal": state["user_goal"],
            "selected_route": state["selected_route"],
            "slots": state["slots"],
            "tool_calls": state["tool_calls"],
            "steps_executed": [s["name"] for s in state["steps"]],
            "needs_user_input": state["needs_user_input"],
        }
        for line in json.dumps(summary, ensure_ascii=False, indent=2).splitlines():
            lines.append(f"  {line}")
        lines.append("")
        lines.append("Final answer:")
        for paragraph in (state["final_answer"] or "").split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=86) or [""])
        lines.append("```")
        lines.append("")
        lines.append(f"**Очікувався маршрут:** `{item['expect']}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_comparison(rows: list[dict]) -> str:
    """Two metrics, because one of them flatters the rule router.

    `agrees` only asks whether the rule route is among what the model chose.
    That counts a hit when the model used two sources and the rules found one
    of them — true, but it hides exactly the weakness worth reporting. `covers`
    is the strict version: did the rules reach everything the model reached.
    """
    agreed = sum(1 for row in rows if row["agrees"])
    covered = sum(1 for row in rows if row["covers"])
    lines = [
        "# Детермінований роутер проти LLM-роутера",
        "",
        "ДЗ №5 доручило вибір інструмента моделі, ДЗ №6 — ключовим словам. "
        "Обидва працюють над тими самими інструментами, тож їх можна порівняти "
        "напряму: ті самі питання, збережені рішення моделі з "
        "`outputs/tool_runs.json`.",
        "",
        f"Згенеровано: {date.today().isoformat()}",
        "",
        f"**Основний маршрут вгадано: {agreed} із {len(rows)}.**  ",
        f"**Покрито всі джерела, які використала модель: {covered} із {len(rows)}.**",
        "",
        "Різниця між цими двома числами і є ціною детермінованого роутера: "
        "він вибирає рівно один маршрут, тоді як модель може взяти два.",
        "",
        "| Питання | LLM обрала | Правила обрали | Основний | Повне покриття |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        llm = ", ".join(f"`{r}`" for r in row["llm_routes"]) or "—"
        lines.append(
            f"| {row['question'][:44]} | {llm} | `{row['rule_route']}` | "
            f"{'так' if row['agrees'] else '**ні**'} | "
            f"{'так' if row['covers'] else '**ні**'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Що показало порівняння",
        "",
        "**Там, де питання містить характерне слово, правила не гірші.** "
        "«Мінімальна заробітна плата», «подай заявку», дата у тексті — "
        "усе це однозначні сигнали, і дешевий роутер ловить їх так само точно, "
        "як модель, за нуль доларів і за мілісекунди замість секунд.",
        "",
        "**Композиція — межа правил.** На питанні «що це таке і скільки зараз» "
        "модель викликала **два** джерела: норму з бази знань і суму з "
        "довідника. Правила обирають рівно один маршрут, тому половина "
        "відповіді втрачається. Щоб це полагодити, довелося б або дозволити "
        "кілька маршрутів одночасно, або повернути модель у маршрутизацію.",
        "",
        "**Правила ламаються на перефразуванні.** Ключові слова покривають те, "
        "що ми передбачили. Питання, сформульоване інакше, падає в "
        "`clarification` — там, де модель зрозуміла б його без підказок.",
        "",
        "## Коли який роутер доречний",
        "",
        "| | Правила | Модель |",
        "|---|---|---|",
        "| Вартість запиту | $0 | ~$0.02–0.04 |",
        "| Затримка | мілісекунди | секунди |",
        "| Відтворюваність | побайтова | ні |",
        "| Пояснюваність | рядок із причиною | треба довіряти |",
        "| Нові формулювання | падають у clarification | зазвичай розуміє |",
        "| Кілька джерел одразу | не вміє | вміє |",
        "",
        "Практичний висновок: це не «або-або». Правила добре працюють як "
        "**швидкий шар для передбачуваних намірів**, а модель — як запасний "
        "варіант для всього іншого. Тоді типові питання коштують нуль, а "
        "нетипові все одно отримують відповідь.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = []
    for scenario in SCENARIOS:
        state = run_flow(scenario["question"])
        results.append({**scenario, "state": state.to_dict()})
        print(
            f"[ok] {scenario['question'][:46]:48s} -> {state.selected_route}"
            + ("  (ask_user)" if state.needs_user_input else "")
        )

    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(render_examples(results), encoding="utf-8")
    print(f"\nWrote {EXAMPLES_PATH.relative_to(ROOT)}")

    if not HW5_RUNS.exists():
        print("outputs/tool_runs.json not found — skipping routing comparison")
        return

    hw5 = json.loads(HW5_RUNS.read_text(encoding="utf-8"))
    rows = []
    for record in hw5["runs"].values():
        llm_routes = list(
            dict.fromkeys(
                TOOL_TO_ROUTE.get(call["tool"], call["tool"])
                for call in record["tool_calls"]
            )
        )
        state = run_flow(record["question"])
        rows.append(
            {
                "question": record["question"],
                "llm_routes": llm_routes,
                "rule_route": state.selected_route,
                "agrees": state.selected_route in llm_routes,
                "covers": set(llm_routes) <= {state.selected_route},
            }
        )

    COMPARISON_PATH.write_text(render_comparison(rows), encoding="utf-8")
    agreed = sum(1 for row in rows if row["agrees"])
    covered = sum(1 for row in rows if row["covers"])
    print(f"Primary route matched: {agreed}/{len(rows)}")
    print(f"All sources covered:   {covered}/{len(rows)}")
    print(f"Wrote {COMPARISON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
