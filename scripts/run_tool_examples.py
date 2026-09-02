"""Run the tool scenarios and render outputs/tool_examples.md.

Five scenarios, each chosen to isolate one routing decision:

    1  amount only          the corpus defines the concept but never the number
    2  norm + amount        both sources needed in one answer
    3  personal arithmetic  depends on the user's dates, not on any document
    4  norm only            the tool must NOT fire — abstention is the result
    5  write + confirm      the two-step gate on an action

Scenario 4 matters as much as the rest: a tool layer that fires on everything
is as broken as one that never fires, and only a case where the right move is
to leave the tools alone can show the difference.

Answers are stored as they arrive so a failed run does not re-spend on the
scenarios already paid for.

Usage:  python scripts/run_tool_examples.py
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import date
from pathlib import Path

from rag_agent import ask

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "outputs" / "tool_runs.json"
REPORT_PATH = ROOT / "outputs" / "tool_examples.md"

SCENARIOS: list[dict] = [
    {
        "id": "s1_amount_only",
        "question": "Яка зараз мінімальна заробітна плата в Україні?",
        "expect": "get_statutory_amount, без search_labour_law",
        "why": (
            "Корпус містить статтю 3 Закону «Про оплату праці» — вона визначає, "
            "**що таке** мінімальна зарплата, і прямо каже, що розмір "
            "встановлюється законом. Самої суми там немає й бути не може: її "
            "щороку задає Закон про Держбюджет. Retrieval тут структурно "
            "безсилий — він поверне визначення й жодної цифри. Це саме той "
            "випадок, коли дані змінюються за розкладом, якого статична база "
            "знань не відстежує."
        ),
    },
    {
        "id": "s2_norm_plus_amount",
        "question": (
            "Що таке мінімальна заробітна плата за законом і скільки вона "
            "становить зараз?"
        ),
        "expect": "search_labour_law + get_statutory_amount",
        "why": (
            "Питання складене з двох half-питань, у яких різні джерела істини. "
            "Норму дає корпус, суму — інструмент. Жодне джерело поодинці не "
            "відповідає повністю, тому це перевірка на те, чи вміє шар "
            "оркестрації комбінувати, а не лише вибирати."
        ),
    },
    {
        "id": "s3_personal_calculation",
        "question": (
            "Я працюю в компанії з 15 березня 2023 року і ще не брав відпустку. "
            "Скільки днів щорічної відпустки я вже накопичив?"
        ),
        "expect": "calculate_vacation_entitlement",
        "why": (
            "Відповідь залежить від дати працевлаштування конкретної людини й "
            "від сьогоднішньої дати. Такого немає в жодному документі — і не "
            "з'явиться, скільки б актів ми не додали. Крім того, LLM ненадійно "
            "рахує різницю дат; винесення арифметики в код робить результат "
            "детермінованим і перевірним."
        ),
    },
    {
        "id": "s4_norm_only",
        "question": "Скільки днів щорічної основної відпустки належить за законом?",
        "expect": "тільки search_labour_law, жодного зовнішнього інструмента",
        "why": (
            "Тут інструмент був би **помилкою**. Відповідь — норма права "
            "(24 календарних дні, стаття 6), вона стабільна й лежить у корпусі. "
            "Сценарій перевіряє зворотний бік маршрутизації: шар, що смикає "
            "інструменти на кожне питання, зламаний так само, як той, що не "
            "смикає їх ніколи."
        ),
    },
    {
        "id": "s5_write_confirmation",
        "question": (
            "Подай заявку на відпустку для працівника emp_001 з 1 жовтня 2026 "
            "року на 10 днів."
        ),
        "expect": "submit_leave_request із confirmed=false, без запису",
        "why": (
            "Це не пошук, а дія: retrieval у принципі не вміє нічого змінювати. "
            "Сценарій перевіряє гейт підтвердження — інструмент зобов'язаний "
            "повернути прев'ю зі статусом requires_confirmation і не записати "
            "нічого, доки користувач не погодився."
        ),
    },
    {
        "id": "s5b_write_confirmed",
        "question": (
            "Так, підтверджую. Подай заявку на відпустку для emp_001 з "
            "1 жовтня 2026 року на 10 днів."
        ),
        "expect": "submit_leave_request із confirmed=true, запис виконано",
        "confirm_writes": True,
        "why": (
            "Друга половина того самого сценарію: після явного підтвердження "
            "той самий інструмент виконує запис і повертає request_id. Разом "
            "із попереднім кроком це показує повний цикл write-дії."
        ),
    },
]


def load_store() -> dict:
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return {"generated_at": date.today().isoformat(), "runs": {}}


def save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def render(store: dict) -> str:
    lines = [
        "# Приклади викликів зовнішніх інструментів",
        "",
        f"Модель: `claude-opus-5` · оркестрація: ручний tool-use цикл "
        f"(`scripts/rag_agent.py`)",
        "",
        f"Згенеровано: {store['generated_at']} · "
        "відтворюється командою `python scripts/run_tool_examples.py`",
        "",
        "Моделі доступні чотири інструменти: `search_labour_law` (база знань "
        "із ДЗ №2) і три зовнішні — `get_statutory_amount`, "
        "`calculate_vacation_entitlement`, `submit_leave_request`. "
        "Який із них викликати — і чи викликати взагалі — вирішує модель.",
        "",
        "---",
        "",
    ]

    for number, scenario in enumerate(SCENARIOS, start=1):
        record = store["runs"].get(scenario["id"])
        if record is None:
            continue

        lines.append(f"## {number}. {scenario['question']}")
        lines.append("")
        lines.append("```")
        lines.append(f"User question: {scenario['question']}")
        lines.append("")

        if not record["tool_calls"]:
            lines.append("Tool called: —  (жодного інструмента не викликано)")
            lines.append("")

        for call in record["tool_calls"]:
            status = "" if call["validated"] else "  [ВІДХИЛЕНО ВАЛІДАЦІЄЮ]"
            lines.append(f"Tool called: {call['tool']}  [{call['kind']}]{status}")
            lines.append(
                f"Input: {json.dumps(call['input'], ensure_ascii=False)}"
            )
            result = json.dumps(call["result"], ensure_ascii=False)
            for offset, line in enumerate(textwrap.wrap(result, width=86)[:6]):
                lines.append(f"{'Result:' if offset == 0 else '       '} {line}")
            lines.append("")

        lines.append("Final answer:")
        for paragraph in record["answer"].split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            lines.extend(textwrap.wrap(paragraph, width=88))
        lines.append("")
        lines.append("Why tool is better than retrieval:")
        lines.extend(textwrap.wrap(scenario["why"], width=88))
        lines.append("```")
        lines.append("")
        lines.append(f"**Очікувалось:** {scenario['expect']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines += ["## Зведення", "", "| # | Сценарій | Викликані інструменти |", "|---|---|---|"]
    for number, scenario in enumerate(SCENARIOS, start=1):
        record = store["runs"].get(scenario["id"])
        if record is None:
            continue
        names = [c["tool"] for c in record["tool_calls"]] or ["—"]
        lines.append(
            f"| {number} | {scenario['question'][:52]} | "
            + ", ".join(f"`{n}`" for n in dict.fromkeys(names))
            + " |"
        )

    total = sum(r["usage"]["cost_usd"] for r in store["runs"].values())
    lines += ["", f"Разом витрачено на прогін: ${total:.4f}."]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    store = {"generated_at": date.today().isoformat(), "runs": {}}
    if not args.rebuild:
        store = load_store()
    store["generated_at"] = date.today().isoformat()

    for scenario in SCENARIOS:
        if scenario["id"] in store["runs"]:
            continue
        result = ask(
            scenario["question"],
            confirm_writes=scenario.get("confirm_writes", False),
        )
        store["runs"][scenario["id"]] = result.to_dict()
        save_store(store)
        names = ", ".join(c.name for c in result.calls) or "—"
        print(f"[ok] {scenario['id']:24s} tools: {names}  ${result.cost_usd:.4f}")

    REPORT_PATH.write_text(render(store), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
