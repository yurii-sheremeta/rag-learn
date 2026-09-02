"""Deterministic agent workflow: route -> plan -> steps -> state -> answer.

HW5 let the model decide which tool to call. This is the same assistant with
that decision moved into code: keyword rules pick a route, the route declares
a fixed sequence of steps, and every step writes what it did into one state
object. No model is involved in routing, and the leave-request route composes
its answer from templates, so a run costs nothing and repeats byte for byte.

Both routers now exist over the same tools, which makes them comparable —
scripts/run_agent_flow.py replays the HW5 questions through this one and
reports where the cheap router and the expensive one disagree.

Two properties this buys that the model-driven version does not have:

    reproducible   same input, same trace, every time
    inspectable    the reason a route was chosen is a stored string, not an
                   inference we would have to take on faith

And one it gives up: anything phrased outside the keyword sets falls through
to clarification, where the model would have understood it.

Usage:
    python scripts/agent_flow.py "яка зараз мінімальна зарплата"
    python scripts/agent_flow.py "подай заявку emp_001 з 01.10.2026 на 10 днів" --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from external_tool import ToolError, call_tool

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

ROUTE_RULES: list[dict[str, Any]] = [
    {
        "route": "leave_request",
        "any_of": ["подай заявку", "оформи заявку", "подати заявку", "оформити відпустку"],
        "description": "Подання заявки на відпустку — дія, що змінює дані",
    },
    {
        "route": "personal_calculation",
        "any_of": ["накопич", "скільки я вже", "мені належить днів", "я працюю з"],
        "description": "Розрахунок за датами конкретного працівника",
    },
    {
        "route": "statutory_amount",
        "any_of": [
            "мінімальна зарплата",
            "мінімальної зарплати",
            "мінімальна заробітна плата",
            "мінімальної заробітної плати",
            "прожитковий мінімум",
            "прожиткового мінімуму",
            "судовий збір",
            "судового збору",
        ],
        "description": "Конкретна сума у гривнях із державного довідника",
    },
    {
        "route": "legal_norm",
        "any_of": [
            "стаття",
            "статті",
            "закон",
            "кзпп",
            "відпустк",
            "звільнен",
            "робочий час",
            "робочого часу",
            "робочий тиждень",
            "оплата праці",
            "охорона праці",
            "колективний договір",
            "вихідн",
            "лікарнян",
        ],
        "description": "Норма права — відповідає база знань",
    },
]

CLARIFICATION_ROUTE = "clarification"

# --------------------------------------------------------------------------
# Slot extraction
# --------------------------------------------------------------------------

EMPLOYEE_RE = re.compile(r"\b(emp_\d{3,6})\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DOTTED_DATE_RE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b")
DAYS_RE = re.compile(r"\b(?:на\s+)?(\d{1,2})\s+(?:календарн\w*\s+)?(?:дн|діб)", re.I)

MONTHS = {
    "січн": 1, "лют": 2, "березн": 3, "квітн": 4, "травн": 5, "червн": 6,
    "липн": 7, "серпн": 8, "вересн": 9, "жовтн": 10, "листопад": 11, "грудн": 12,
}
WORDY_DATE_RE = re.compile(r"\b(\d{1,2})\s+([а-яіїєґ]+)\s+(\d{4})", re.IGNORECASE)

AMOUNT_KEYWORDS = {
    "minimum_wage_hourly": ["погодин", "за годину"],
    "subsistence_minimum_able_bodied": ["прожитков"],
    "court_fee_labour_claim": ["судовий збір", "судового збору", "мито"],
    "minimum_wage_monthly": ["мінімальн"],
}


def extract_dates(text: str) -> list[str]:
    """Every date in the question, normalised to ISO, in order of appearance."""
    found: list[tuple[int, str]] = []

    for match in ISO_DATE_RE.finditer(text):
        found.append((match.start(), match.group(0)))

    for match in DOTTED_DATE_RE.finditer(text):
        day, month, year = match.groups()
        found.append((match.start(), f"{year}-{int(month):02d}-{int(day):02d}"))

    for match in WORDY_DATE_RE.finditer(text):
        day, month_word, year = match.groups()
        for stem, number in MONTHS.items():
            if month_word.lower().startswith(stem):
                found.append((match.start(), f"{year}-{number:02d}-{int(day):02d}"))
                break

    return [value for _, value in sorted(found)]


def extract_slots(question: str) -> dict[str, Any]:
    """Pull the structured bits a route may need out of free text."""
    lowered = question.lower()
    slots: dict[str, Any] = {}

    employee = EMPLOYEE_RE.search(question)
    if employee:
        slots["employee_id"] = employee.group(1).lower()

    dates = extract_dates(question)
    if dates:
        slots["dates"] = dates

    days = DAYS_RE.search(question)
    if days:
        slots["days"] = int(days.group(1))

    for indicator, keywords in AMOUNT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            slots["indicator"] = indicator
            break

    return slots


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One executed step: what ran, on what, and what came back."""

    name: str
    tool: str | None
    tool_input: dict[str, Any] | None
    observation: Any
    ok: bool = True


@dataclass
class FlowState:
    """Everything the workflow knows. Carried across steps, returned to caller."""

    user_goal: str
    selected_route: str | None = None
    route_reason: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    needs_user_input: str | None = None
    final_answer: str | None = None

    @property
    def tool_calls(self) -> list[str]:
        return [step.tool for step in self.steps if step.tool]

    def record(
        self,
        name: str,
        observation: Any,
        tool: str | None = None,
        tool_input: dict | None = None,
        ok: bool = True,
    ) -> Any:
        """Append a step and its observation. Returns the observation."""
        self.steps.append(Step(name, tool, tool_input, observation, ok))
        self.observations.append(observation)
        return observation

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tool_calls"] = self.tool_calls
        return payload


# --------------------------------------------------------------------------
# Step 1 - routing
# --------------------------------------------------------------------------


def route(question: str) -> tuple[str, str]:
    """Pick a route by keyword. Returns (route, human-readable reason).

    Order matters: the rules are checked most specific first, so that
    "подай заявку на відпустку" lands on leave_request rather than on
    legal_norm, which also matches "відпустк".
    """
    lowered = question.lower()
    for rule in ROUTE_RULES:
        for keyword in rule["any_of"]:
            if keyword in lowered:
                return rule["route"], f"збіг за ключем «{keyword}» → {rule['description']}"
    return CLARIFICATION_ROUTE, "жодне правило не спрацювало"


PLANS: dict[str, list[str]] = {
    "statutory_amount": ["extract_slots", "lookup_amount", "compose_answer"],
    "legal_norm": ["extract_slots", "search_knowledge_base", "compose_answer"],
    "personal_calculation": [
        "extract_slots",
        "check_required_slots",
        "calculate_entitlement",
        "compose_answer",
    ],
    "leave_request": [
        "extract_slots",
        "check_required_slots",
        "calculate_entitlement",
        "preview_request",
        "compose_answer",
    ],
    CLARIFICATION_ROUTE: ["compose_clarification"],
}


# --------------------------------------------------------------------------
# Mock tool: knowledge base lookup
# --------------------------------------------------------------------------

_retriever = None


def search_knowledge_base(query: str, k: int = 2) -> dict[str, Any]:
    """Knowledge-base lookup. Falls back to a fixture when the index is absent.

    The fallback exists so the whole workflow stays runnable with no model
    downloads at all — the assignment asks for mock tools, and a flow that
    cannot be demonstrated without 1.1 GB of weights is a worse deliverable.
    """
    global _retriever
    try:
        if _retriever is None:
            from retrieval import Retriever

            _retriever = Retriever()
        hits = _retriever.search(query, k=k)
        return {
            "source": "faiss_index",
            "chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 4),
                    "section": hit.metadata["section"],
                    "source_file": hit.metadata["source_file"],
                    "text": hit.text.split("\n\n", 1)[-1][:400],
                }
                for hit in hits
            ],
        }
    except Exception as error:
        return {
            "source": "fixture",
            "note": f"індекс недоступний ({type(error).__name__}), використано заглушку",
            "chunks": [
                {
                    "chunk_id": "zakon_pro_vidpustky_chunk_0006",
                    "score": 0.87,
                    "section": "Стаття 6. Щорічна основна відпустка та її тривалість",
                    "source_file": "data/raw/zakon_pro_vidpustky.html",
                    "text": (
                        "Щорічна основна відпустка надається працівникам тривалістю "
                        "не менш як 24 календарних дні за відпрацьований робочий рік."
                    ),
                }
            ],
        }


# --------------------------------------------------------------------------
# Step execution
# --------------------------------------------------------------------------


def _tool_step(
    state: FlowState, name: str, tool: str, arguments: dict[str, Any]
) -> Any:
    """Run a registered external tool, recording success or the error."""
    try:
        result = call_tool(tool, arguments)
        return state.record(name, result, tool=tool, tool_input=arguments)
    except ToolError as error:
        return state.record(
            name, {"error": str(error)}, tool=tool, tool_input=arguments, ok=False
        )


def run_flow(question: str, confirmed: bool = False) -> FlowState:
    """Route the goal, run the route's plan, and return the full state."""
    state = FlowState(user_goal=question)
    state.selected_route, state.route_reason = route(question)
    state.plan = list(PLANS[state.selected_route])

    if state.selected_route == CLARIFICATION_ROUTE:
        state.needs_user_input = (
            "Уточніть, будь ласка: вас цікавить норма закону, конкретна сума "
            "(мінімальна зарплата, прожитковий мінімум), розрахунок вашої "
            "відпустки чи подання заявки?"
        )
        state.record("compose_clarification", state.needs_user_input)
        state.final_answer = state.needs_user_input
        return state

    state.slots = state.record("extract_slots", extract_slots(question))

    if state.selected_route == "statutory_amount":
        indicator = state.slots.get("indicator", "minimum_wage_monthly")
        on_date = (state.slots.get("dates") or [None])[0]
        result = _tool_step(
            state,
            "lookup_amount",
            "get_statutory_amount",
            {"indicator": indicator, "on_date": on_date},
        )
        state.final_answer = _answer_amount(result)
        return state

    if state.selected_route == "legal_norm":
        result = state.record(
            "search_knowledge_base",
            search_knowledge_base(question),
            tool="search_knowledge_base",
            tool_input={"query": question},
        )
        state.final_answer = _answer_norm(result)
        return state

    # Both remaining routes need an employment start date; leave_request also
    # needs an employee. Missing slots are a route transition, not a crash.
    dates = state.slots.get("dates", [])
    if state.selected_route == "leave_request" and "employee_id" not in state.slots:
        return _ask_for(state, "ідентифікатор працівника у форматі emp_001")
    if state.selected_route == "personal_calculation" and not dates:
        return _ask_for(state, "дату, з якої ви працюєте в компанії")

    if state.selected_route == "personal_calculation":
        state.record("check_required_slots", {"employment_start_date": dates[0]})
        result = _tool_step(
            state,
            "calculate_entitlement",
            "calculate_vacation_entitlement",
            {"employment_start_date": dates[0]},
        )
        state.final_answer = _answer_calculation(result)
        return state

    # leave_request
    from external_tool import MOCK_EMPLOYEES

    employee_id = state.slots["employee_id"]
    employee = MOCK_EMPLOYEES.get(employee_id)
    if employee is None:
        return _ask_for(state, f"чинний ідентифікатор — {employee_id} не знайдено")

    start_date = dates[0] if dates else None
    days = state.slots.get("days")
    if not start_date or not days:
        return _ask_for(state, "дату початку відпустки та кількість днів")

    state.record(
        "check_required_slots",
        {"employee_id": employee_id, "start_date": start_date, "days": days},
    )

    # Step that uses the PREVIOUS step's state: entitlement is computed from
    # the employee found above, then compared with the days asked for.
    entitlement = _tool_step(
        state,
        "calculate_entitlement",
        "calculate_vacation_entitlement",
        {
            "employment_start_date": employee["employment_start_date"],
            "as_of_date": start_date,
        },
    )

    preview = _tool_step(
        state,
        "preview_request",
        "submit_leave_request",
        {
            "employee_id": employee_id,
            "start_date": start_date,
            "days": days,
            "confirmed": bool(confirmed),
        },
    )

    state.final_answer = _answer_leave(preview, entitlement, confirmed)
    if not confirmed:
        state.needs_user_input = "Підтвердіть подання заявки."
    return state


def _ask_for(state: FlowState, what: str) -> FlowState:
    """Route transition: a route was picked but its slots are incomplete."""
    message = f"Щоб продовжити, вкажіть, будь ласка, {what}."
    state.record("check_required_slots", {"missing": what}, ok=False)
    state.needs_user_input = message
    state.final_answer = message
    state.plan.append("ask_user")
    return state


# --------------------------------------------------------------------------
# Answer composition - templates, no model
# --------------------------------------------------------------------------


def _answer_amount(result: dict) -> str:
    if "error" in result:
        return f"Не вдалося отримати значення: {result['error']}"
    lines = [
        f"{result['label']} станом на {result['on_date']} — "
        f"{result['value']} {result['unit']}.",
        f"Діє з {result['effective_from']}. Підстава: {result['legal_basis']}.",
        f"[get_statutory_amount]",
    ]
    if result.get("data_status") == "fixture":
        lines.append(
            "Увага: значення демонстраційне і не звірене з чинним "
            "Законом про Держбюджет."
        )
    return "\n".join(lines)


def _answer_norm(result: dict) -> str:
    chunk = result["chunks"][0]
    note = "" if result["source"] == "faiss_index" else f" ({result['note']})"
    return (
        f"{chunk['text']}\n\n"
        f"[{chunk['section']}, {chunk['chunk_id']}]{note}\n"
        f"Джерело: {chunk['source_file']}"
    )


def _answer_calculation(result: dict) -> str:
    if "error" in result:
        return f"Не вдалося порахувати: {result['error']}"
    eligible = (
        "Право на повну відпустку вже настало."
        if result["eligible_for_full_leave"]
        else "Права на повну відпустку ще немає — потрібно шість місяців роботи."
    )
    return (
        f"Станом на {result['as_of_date']} ви відпрацювали "
        f"{result['days_worked']} днів (близько {result['months_worked']} місяців).\n"
        f"Накопичено {result['accrued_days']} з {result['annual_days']} днів "
        f"щорічної відпустки, залишок {result['remaining_days']}.\n"
        f"{eligible}\n"
        f"[calculate_vacation_entitlement] Підстава: {result['legal_basis']}"
    )


def _answer_leave(preview: dict, entitlement: dict, confirmed: bool) -> str:
    if "error" in preview:
        return f"Заявку не подано: {preview['error']}"

    warning = ""
    if preview.get("exceeds_accrued"):
        warning = (
            f"\nУвага: запитано {preview['days']} днів, а накопичено лише "
            f"{entitlement.get('accrued_days')}."
        )

    if preview.get("written"):
        return (
            f"Заявку подано. Номер {preview['request_id']}, "
            f"{preview['employee_name']} ({preview['employee_id']}), "
            f"з {preview['start_date']} на {preview['days']} днів."
            f"{warning}\n[submit_leave_request]"
        )

    return (
        f"Готую заявку: {preview['employee_name']} ({preview['employee_id']}), "
        f"з {preview['start_date']} на {preview['days']} днів. "
        f"Накопичено на цю дату: {preview['accrued_days_at_start']} днів."
        f"{warning}\n"
        f"Заявку ще НЕ подано — підтвердіть, і я її надішлю.\n"
        f"[submit_leave_request, status={preview['status']}]"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def print_state(state: FlowState) -> None:
    print(f"\nQuestion: {state.user_goal}")
    print("=" * 78)
    print(f"Route: {state.selected_route}")
    print(f"  reason: {state.route_reason}")
    print(f"Plan: {' → '.join(state.plan)}")
    print(f"Slots: {json.dumps(state.slots, ensure_ascii=False)}")

    for number, step in enumerate(state.steps, start=1):
        marker = "" if step.ok else "  [НЕ ПРОЙДЕНО]"
        print(f"\nStep {number}: {step.name}{marker}")
        if step.tool:
            print(f"  Tool: {step.tool}")
            print(f"  Input: {json.dumps(step.tool_input, ensure_ascii=False)}")
        observation = json.dumps(step.observation, ensure_ascii=False, default=str)
        print(f"  Observation: {observation[:200]}{'…' if len(observation) > 200 else ''}")

    print("\nFinal answer:")
    for paragraph in (state.final_answer or "").split("\n"):
        for line in textwrap.wrap(paragraph, width=76) or [""]:
            print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic agent workflow")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--confirm", action="store_true", help="approve a write step")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.question:
        parser.error("provide a question")

    state = run_flow(" ".join(args.question), confirmed=args.confirm)
    if args.json:
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print_state(state)


if __name__ == "__main__":
    sys.exit(main())
