"""The HW6 workflow rebuilt as a LangGraph state graph.

Deliberately a port, not a rewrite: routing rules, slot extraction, tools and
answer templates are imported from agent_flow rather than copied. Only the
orchestration changes — a hand-written sequence of ifs becomes nodes and edges.
Keeping the logic shared is what makes the two comparable: run_langgraph.py
diffs the two implementations on the same questions, and a difference there
means the graph is wrong, not that the rules changed.

Why LangGraph over the alternatives: the HW6 flow already had exactly the
three things it models — a typed state passed between steps, discrete steps,
and branch decisions. LlamaIndex Workflow is event-driven, which is a poorer
fit for a linear route; CrewAI is built around multiple collaborating agents,
which this is not. Choosing the framework whose primitives already match the
shape of the problem is the whole point of the exercise.

Graph shape:

    classify_request
        │  conditional edge on selected_route
        ├─ statutory_amount ──► extract → lookup_amount ─────┐
        ├─ legal_norm ────────► extract → search_kb ─────────┤
        ├─ personal_calculation► extract → check ──┬─────────┤
        ├─ leave_request ──────► extract → check ──┤         ├─► build_answer ─► END
        │                                          │  conditional edge on
        │                                          │  whether slots are complete
        │                                          └─► ask_user ──────────────► END
        └─ clarification ─────────────────────────────────────► END

Usage:
    python scripts/langgraph_flow.py "яка зараз мінімальна зарплата"
    python scripts/langgraph_flow.py "..." --json
    python scripts/langgraph_flow.py --diagram
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent_flow import (
    CLARIFICATION_ROUTE,
    _answer_amount,
    _answer_calculation,
    _answer_leave,
    _answer_norm,
    extract_slots,
    route,
    search_knowledge_base,
)
from external_tool import MOCK_EMPLOYEES, ToolError, call_tool


class AgentState(TypedDict, total=False):
    """State handed from node to node. Every node returns a partial update.

    Mirrors FlowState from agent_flow one field at a time, so the two runs can
    be diffed field by field rather than eyeballed.
    """

    user_question: str
    selected_route: str
    route_reason: str
    slots: dict[str, Any]
    tool_result: dict[str, Any]
    entitlement: dict[str, Any]
    observations: list[dict[str, Any]]
    nodes_executed: list[str]
    needs_user_input: str | None
    final_answer: str


def _visit(state: AgentState, name: str) -> list[str]:
    """Node names in execution order — the graph's own trace."""
    return [*state.get("nodes_executed", []), name]


def _run_tool(state: AgentState, tool: str, arguments: dict) -> dict:
    """Call a registered tool, turning validation failures into a result dict."""
    try:
        result = call_tool(tool, arguments)
    except ToolError as error:
        result = {"error": str(error)}
    state.setdefault("observations", []).append(
        {"tool": tool, "input": arguments, "result": result}
    )
    return result


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def classify_request(state: AgentState) -> AgentState:
    selected, reason = route(state["user_question"])
    return {
        "selected_route": selected,
        "route_reason": reason,
        "nodes_executed": _visit(state, "classify_request"),
        "observations": [],
    }


def extract_request_slots(state: AgentState) -> AgentState:
    return {
        "slots": extract_slots(state["user_question"]),
        "nodes_executed": _visit(state, "extract_slots"),
    }


def lookup_amount(state: AgentState) -> AgentState:
    slots = state["slots"]
    result = _run_tool(
        state,
        "get_statutory_amount",
        {
            "indicator": slots.get("indicator", "minimum_wage_monthly"),
            "on_date": (slots.get("dates") or [None])[0],
        },
    )
    return {
        "tool_result": result,
        "observations": state["observations"],
        "nodes_executed": _visit(state, "lookup_amount"),
    }


def search_kb(state: AgentState) -> AgentState:
    result = search_knowledge_base(state["user_question"])
    state.setdefault("observations", []).append(
        {"tool": "search_knowledge_base", "input": {"query": state["user_question"]}, "result": result}
    )
    return {
        "tool_result": result,
        "observations": state["observations"],
        "nodes_executed": _visit(state, "search_knowledge_base"),
    }


def check_required_slots(state: AgentState) -> AgentState:
    """Decide whether the chosen route has everything it needs to run."""
    slots = state["slots"]
    dates = slots.get("dates", [])
    missing: str | None = None

    if state["selected_route"] == "personal_calculation" and not dates:
        missing = "дату, з якої ви працюєте в компанії"
    elif state["selected_route"] == "leave_request":
        if "employee_id" not in slots:
            missing = "ідентифікатор працівника у форматі emp_001"
        elif slots["employee_id"] not in MOCK_EMPLOYEES:
            missing = f"чинний ідентифікатор — {slots['employee_id']} не знайдено"
        elif not dates or "days" not in slots:
            missing = "дату початку відпустки та кількість днів"

    return {
        "needs_user_input": (
            f"Щоб продовжити, вкажіть, будь ласка, {missing}." if missing else None
        ),
        "nodes_executed": _visit(state, "check_required_slots"),
    }


def calculate_entitlement(state: AgentState) -> AgentState:
    slots = state["slots"]
    if state["selected_route"] == "leave_request":
        employee = MOCK_EMPLOYEES[slots["employee_id"]]
        arguments = {
            "employment_start_date": employee["employment_start_date"],
            "as_of_date": slots["dates"][0],
        }
    else:
        arguments = {"employment_start_date": slots["dates"][0]}

    result = _run_tool(state, "calculate_vacation_entitlement", arguments)
    return {
        "entitlement": result,
        "tool_result": result,
        "observations": state["observations"],
        "nodes_executed": _visit(state, "calculate_entitlement"),
    }


def preview_request(state: AgentState) -> AgentState:
    slots = state["slots"]
    result = _run_tool(
        state,
        "submit_leave_request",
        {
            "employee_id": slots["employee_id"],
            "start_date": slots["dates"][0],
            "days": slots["days"],
            "confirmed": False,
        },
    )
    return {
        "tool_result": result,
        "observations": state["observations"],
        "nodes_executed": _visit(state, "preview_request"),
    }


def build_answer(state: AgentState) -> AgentState:
    """Single exit node: every successful route composes its answer here."""
    selected = state["selected_route"]
    result = state.get("tool_result", {})

    if selected == "statutory_amount":
        answer = _answer_amount(result)
    elif selected == "legal_norm":
        answer = _answer_norm(result)
    elif selected == "personal_calculation":
        answer = _answer_calculation(result)
    else:
        answer = _answer_leave(result, state.get("entitlement", {}), confirmed=False)

    return {
        "final_answer": answer,
        "needs_user_input": (
            "Підтвердіть подання заявки." if selected == "leave_request" else None
        ),
        "nodes_executed": _visit(state, "build_answer"),
    }


def ask_user(state: AgentState) -> AgentState:
    """Terminal node for a route that was picked but cannot proceed."""
    return {
        "final_answer": state["needs_user_input"],
        "nodes_executed": _visit(state, "ask_user"),
    }


def ask_clarification(state: AgentState) -> AgentState:
    message = (
        "Уточніть, будь ласка: вас цікавить норма закону, конкретна сума "
        "(мінімальна зарплата, прожитковий мінімум), розрахунок вашої "
        "відпустки чи подання заявки?"
    )
    return {
        "needs_user_input": message,
        "final_answer": message,
        "nodes_executed": _visit(state, "ask_clarification"),
    }


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def route_decision(state: AgentState) -> str:
    """First conditional edge: which branch the request enters."""
    return state["selected_route"]


def slots_decision(state: AgentState) -> str:
    """Second conditional edge: proceed, or stop and ask the user."""
    return "ask_user" if state.get("needs_user_input") else "continue"


def after_calculation(state: AgentState) -> str:
    """Leave requests need one more tool call before the answer is composed."""
    return "preview" if state["selected_route"] == "leave_request" else "answer"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_request", classify_request)
    graph.add_node("extract_slots", extract_request_slots)
    graph.add_node("check_required_slots", check_required_slots)
    graph.add_node("lookup_amount", lookup_amount)
    graph.add_node("search_knowledge_base", search_kb)
    graph.add_node("calculate_entitlement", calculate_entitlement)
    graph.add_node("preview_request", preview_request)
    graph.add_node("build_answer", build_answer)
    graph.add_node("ask_user", ask_user)
    graph.add_node("ask_clarification", ask_clarification)

    graph.set_entry_point("classify_request")

    # Conditional edge 1 - five-way branch on the route.
    graph.add_conditional_edges(
        "classify_request",
        route_decision,
        {
            "statutory_amount": "extract_slots",
            "legal_norm": "extract_slots",
            "personal_calculation": "extract_slots",
            "leave_request": "extract_slots",
            CLARIFICATION_ROUTE: "ask_clarification",
        },
    )

    # After slots are extracted, routes that need no validation go straight to
    # their tool; the two that depend on user-supplied data get checked first.
    graph.add_conditional_edges(
        "extract_slots",
        route_decision,
        {
            "statutory_amount": "lookup_amount",
            "legal_norm": "search_knowledge_base",
            "personal_calculation": "check_required_slots",
            "leave_request": "check_required_slots",
        },
    )

    # Conditional edge 2 - the interesting one: incomplete input is a third
    # outcome, neither success nor error.
    graph.add_conditional_edges(
        "check_required_slots",
        slots_decision,
        {"continue": "calculate_entitlement", "ask_user": "ask_user"},
    )

    graph.add_conditional_edges(
        "calculate_entitlement",
        after_calculation,
        {"preview": "preview_request", "answer": "build_answer"},
    )

    graph.add_edge("lookup_amount", "build_answer")
    graph.add_edge("search_knowledge_base", "build_answer")
    graph.add_edge("preview_request", "build_answer")
    graph.add_edge("build_answer", END)
    graph.add_edge("ask_user", END)
    graph.add_edge("ask_clarification", END)

    return graph.compile()


APP = build_graph()


def run(question: str) -> AgentState:
    return APP.invoke(
        {
            "user_question": question,
            "nodes_executed": [],
            "observations": [],
            "slots": {},
        }
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def print_state(state: AgentState) -> None:
    print(f"\nInput: {state['user_question']}")
    print("=" * 78)
    print(f"Route: {state['selected_route']}")
    print(f"  reason: {state.get('route_reason')}")
    print(f"Nodes executed: {' → '.join(state['nodes_executed'])}")
    print(f"Slots: {json.dumps(state.get('slots', {}), ensure_ascii=False)}")

    for observation in state.get("observations", []):
        print(f"\n  Tool: {observation['tool']}")
        print(f"  Input: {json.dumps(observation['input'], ensure_ascii=False)}")
        result = json.dumps(observation["result"], ensure_ascii=False)
        print(f"  Result: {result[:180]}{'…' if len(result) > 180 else ''}")

    print("\nFinal answer:")
    for paragraph in (state.get("final_answer") or "").split("\n"):
        for line in textwrap.wrap(paragraph, width=76) or [""]:
            print(f"  {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph port of the agent workflow")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--diagram", action="store_true", help="print the graph as Mermaid")
    args = parser.parse_args()

    if args.diagram:
        print(APP.get_graph().draw_mermaid())
        return

    if not args.question:
        parser.error("provide a question or --diagram")

    state = run(" ".join(args.question))
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    else:
        print_state(state)


if __name__ == "__main__":
    sys.exit(main())
