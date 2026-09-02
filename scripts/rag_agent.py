"""Orchestration layer: Claude picks between the knowledge base and the tools.

RAG is exposed as just another tool (`search_labour_law`) alongside the three
external ones, so routing is a single decision the model makes with all options
on the table. That also makes the interesting failure visible: calling a tool
where retrieval was right, or answering from a document where only live data
would do.

The loop is written by hand rather than using the SDK's beta tool runner. Two
reasons: the assignment is about showing the integration pattern, and every
step — the model's raw arguments, whether validation accepted them, what the
source returned — has to be captured for the report. A hand-written loop makes
that trace the natural output rather than something recovered afterwards.

Usage:
    python scripts/rag_agent.py "яка зараз мінімальна зарплата"
    python scripts/rag_agent.py "..." --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from external_tool import ToolError, anthropic_tool_specs, call_tool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5"
MAX_TOKENS = 8000
MAX_ITERATIONS = 6
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SEARCH_TOOL = {
    "name": "search_labour_law",
    "description": (
        "Шукає у базі знань із шести актів трудового права України: КЗпП, "
        "закони про відпустки, оплату праці, охорону праці, колективні "
        "договори, трудові відносини в умовах воєнного стану. "
        "ВИКЛИКАТИ для питань про норми, права, обов'язки, визначення понять "
        "і порядок дій. "
        "НЕ ВИКЛИКАТИ, коли потрібна конкретна грошова сума або розрахунок "
        "за датами конкретного працівника — цього в актах немає."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Пошуковий запит українською.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

SYSTEM = """Ти асистент з трудового права України для працівників.

У тебе є база знань із шести нормативних актів і три зовнішні інструменти.

ЯК ОБИРАТИ ДЖЕРЕЛО
- Норма права, визначення, порядок дій, права та обов'язки → search_labour_law.
- Конкретна сума у гривнях (мінімальна зарплата, прожитковий мінімум,
  судовий збір) → get_statutory_amount. Ці суми щороку встановлює закон про
  держбюджет, якого в базі знань немає.
- Розрахунок за датами конкретної людини → calculate_vacation_entitlement.
- Якщо питання складається з двох частин — «що це таке» і «скільки це зараз» —
  виклич обидва джерела: норму з бази знань, суму з інструмента.

ДІЇ, ЩО ЗМІНЮЮТЬ ДАНІ
submit_leave_request записує заявку. Спершу викликай його з confirmed=false,
покажи користувачу параметри й попроси підтвердити. Ніколи не став
confirmed=true самостійно, доки користувач явно не погодився.

ЯК ВІДПОВІДАТИ
- Відповідай стисло, простою мовою.
- Норму з бази знань цитуй у форматі [Стаття N Акт, chunk_id].
- Дані з інструмента підписуй назвою інструмента: [get_statutory_amount].
- Якщо інструмент повернув data_status "fixture" — обов'язково попередь
  користувача, що це демонстраційні, а не звірені дані.
- Якщо жодне джерело не дало відповіді, скажи прямо, що інформації немає.
- Не використовуй знання поза наданими джерелами.
"""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any
    kind: str = "external"

    def to_dict(self) -> dict:
        return {
            "tool": self.name,
            "kind": self.kind,
            "input": self.arguments,
            "validated": self.ok,
            "result": self.result,
        }


@dataclass
class AgentAnswer:
    question: str
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * 5.0 + self.output_tokens / 1e6 * 25.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "tool_calls": [call.to_dict() for call in self.calls],
            "iterations": self.iterations,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 5),
            },
        }


_retriever = None


def search_labour_law(query: str, k: int = 3) -> dict[str, Any]:
    """The HW2 retriever, wrapped so the model can reach it as a tool."""
    global _retriever
    if _retriever is None:
        from retrieval import Retriever

        _retriever = Retriever()

    hits = _retriever.search(query, k=k)
    return {
        "query": query,
        "chunks": [
            {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "section": hit.metadata["section"],
                "source_file": hit.metadata["source_file"],
                "text": hit.text,
            }
            for hit in hits
        ],
    }


def execute(name: str, arguments: dict[str, Any]) -> ToolCall:
    """Run one tool call, converting validation failures into error results."""
    if name == "search_labour_law":
        try:
            return ToolCall(name, arguments, True, search_labour_law(**arguments), "rag")
        except Exception as error:
            return ToolCall(name, arguments, False, {"error": str(error)}, "rag")

    try:
        return ToolCall(name, arguments, True, call_tool(name, arguments))
    except ToolError as error:
        return ToolCall(name, arguments, False, {"error": str(error)})
    except Exception as error:
        return ToolCall(
            name, arguments, False, {"error": f"{type(error).__name__}: {error}"}
        )


def ask(question: str, confirm_writes: bool = False) -> AgentAnswer:
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic()
    tools = [SEARCH_TOOL, *anthropic_tool_specs()]
    system = SYSTEM
    if confirm_writes:
        system += (
            "\nКОРИСТУВАЧ УЖЕ ПІДТВЕРДИВ ЗАПИС. Якщо потрібно подати заявку, "
            "після показу параметрів виклич submit_leave_request із "
            "confirmed=true.\n"
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentAnswer(question=question, text="")

    for iteration in range(1, MAX_ITERATIONS + 1):
        request = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "thinking": {"type": "adaptive"},
            "tools": tools,
            "messages": messages,
        }
        try:
            response = client.beta.messages.create(
                **request, betas=[FALLBACK_BETA], fallbacks="default"
            )
        except Exception:
            response = client.messages.create(**request)

        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens
        result.iterations = iteration

        if response.stop_reason != "tool_use":
            result.text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            call = execute(block.name, dict(block.input))
            result.calls.append(call)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(call.result, ensure_ascii=False),
                    **({} if call.ok else {"is_error": True}),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return result


def print_answer(result: AgentAnswer) -> None:
    print(f"\nQuestion: {result.question}")
    print("=" * 78)
    for call in result.calls:
        status = "ok" if call.ok else "REJECTED"
        print(f"\nTool called: {call.name}  [{call.kind}, {status}]")
        print(f"  Input:  {json.dumps(call.arguments, ensure_ascii=False)}")
        summary = json.dumps(call.result, ensure_ascii=False)
        print(f"  Result: {summary[:220]}{'…' if len(summary) > 220 else ''}")

    print("\nAnswer:")
    for paragraph in result.text.split("\n"):
        if not paragraph.strip():
            print()
            continue
        for line in textwrap.wrap(paragraph, width=76):
            print(f"  {line}")
    print(
        f"\niterations {result.iterations} | "
        f"tokens {result.input_tokens}+{result.output_tokens} | "
        f"${result.cost_usd:.4f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-using labour law assistant")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--confirm-writes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.question:
        parser.error("provide a question")

    result = ask(" ".join(args.question), confirm_writes=args.confirm_writes)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_answer(result)


if __name__ == "__main__":
    sys.exit(main())
