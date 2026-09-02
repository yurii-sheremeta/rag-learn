"""External tools the labour-law assistant can call, with validation.

The knowledge base is six acts of Ukrainian labour law. Those acts define what
the minimum wage IS, but the amount itself is set every year by a separate
budget law that the corpus does not and structurally cannot contain. HW4 showed
this directly: asked about payroll taxes and court fees, the RAG pipeline
correctly refused, because refusing was the only honest answer available to it.

That gap is what these tools close. Three of them, chosen to cover the three
reasons a tool beats retrieval:

    get_statutory_amount          data that changes on a schedule the corpus
                                  cannot track
    calculate_vacation_entitlement  arithmetic over the user's own dates, which
                                  no document can contain
    submit_leave_request          an action, not a lookup

Validation is Pydantic, not hand-written ifs: the schema is the contract, and
the same model both documents the tool to Claude and rejects bad input at the
boundary. None of the tools accepts free-form text that reaches a query engine
— every parameter is a typed scalar, a date, or a closed enum, so there is no
raw-SQL surface for the model to fill in.

Usage:
    python scripts/external_tool.py --list
    python scripts/external_tool.py get_statutory_amount '{"indicator": "minimum_wage_monthly"}'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "reference" / "statutory_amounts.json"
LEAVE_REQUESTS_PATH = ROOT / "data" / "reference" / "leave_requests.json"

EMPLOYEE_ID_RE = re.compile(r"^emp_\d{3,6}$")

MOCK_EMPLOYEES: dict[str, dict[str, str]] = {
    "emp_001": {"name": "Олена Ткаченко", "employment_start_date": "2023-03-15"},
    "emp_002": {"name": "Андрій Бондаренко", "employment_start_date": "2026-06-01"},
}


class ToolError(Exception):
    """Raised when input fails validation, so the caller can mark is_error."""


# --------------------------------------------------------------------------
# Tool 1 - get_statutory_amount (read)
# --------------------------------------------------------------------------

Indicator = Literal[
    "minimum_wage_monthly",
    "minimum_wage_hourly",
    "subsistence_minimum_able_bodied",
    "court_fee_labour_claim",
]


class StatutoryAmountInput(BaseModel):
    """Input contract for get_statutory_amount."""

    indicator: Indicator = Field(
        description="Which statutory value to look up. Closed set — the model "
        "cannot invent an indicator name."
    )
    on_date: date | None = Field(
        default=None,
        description="Date the value should be valid on, ISO 8601. Defaults to today.",
    )

    @field_validator("on_date")
    @classmethod
    def not_absurdly_old(cls, value: date | None) -> date | None:
        if value is not None and value.year < 2020:
            raise ValueError("on_date before 2020 is outside the reference data")
        return value


def get_statutory_amount(indicator: str, on_date: str | None = None) -> dict[str, Any]:
    """Return the statutory monetary value in force on a given date."""
    try:
        payload = StatutoryAmountInput(indicator=indicator, on_date=on_date)
    except ValidationError as error:
        raise ToolError(_format_validation_error(error)) from error

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    entry = reference["indicators"][payload.indicator]
    target = payload.on_date or date.today()

    for period in entry["periods"]:
        starts = date.fromisoformat(period["effective_from"])
        ends = date.fromisoformat(period["effective_to"]) if period["effective_to"] else None
        if starts <= target and (ends is None or target <= ends):
            return {
                "indicator": payload.indicator,
                "label": entry["label"],
                "value": period["value"],
                "unit": entry["unit"],
                "effective_from": period["effective_from"],
                "effective_to": period["effective_to"],
                "on_date": target.isoformat(),
                "legal_basis": entry["legal_basis"],
                "data_status": reference["data_status"],
                "disclaimer": reference["disclaimer"],
            }

    raise ToolError(
        f"no value for {payload.indicator} on {target.isoformat()}; "
        f"reference data starts at {entry['periods'][0]['effective_from']}"
    )


# --------------------------------------------------------------------------
# Tool 2 - calculate_vacation_entitlement (read / compute)
# --------------------------------------------------------------------------


class VacationInput(BaseModel):
    """Input contract for calculate_vacation_entitlement."""

    employment_start_date: date = Field(
        description="First day of employment, ISO 8601."
    )
    as_of_date: date | None = Field(
        default=None, description="Date to calculate at, ISO 8601. Defaults to today."
    )
    annual_days: int = Field(
        default=24,
        ge=24,
        le=59,
        description="Annual entitlement in calendar days. Article 6 sets the floor "
        "at 24; higher values apply to specific categories.",
    )
    used_days: int = Field(
        default=0, ge=0, le=365, description="Days already taken this working year."
    )

    @field_validator("employment_start_date")
    @classmethod
    def not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("employment_start_date is in the future")
        return value


def calculate_vacation_entitlement(
    employment_start_date: str,
    as_of_date: str | None = None,
    annual_days: int = 24,
    used_days: int = 0,
) -> dict[str, Any]:
    """Accrued and remaining annual leave for one person on one date."""
    try:
        payload = VacationInput(
            employment_start_date=employment_start_date,
            as_of_date=as_of_date,
            annual_days=annual_days,
            used_days=used_days,
        )
    except ValidationError as error:
        raise ToolError(_format_validation_error(error)) from error

    target = payload.as_of_date or date.today()
    if target < payload.employment_start_date:
        raise ToolError("as_of_date is earlier than employment_start_date")

    days_worked = (target - payload.employment_start_date).days
    months_worked = days_worked // 30
    accrued = min(
        payload.annual_days,
        round(payload.annual_days * days_worked / 365, 1),
    )
    remaining = round(max(0.0, accrued - payload.used_days), 1)

    return {
        "employment_start_date": payload.employment_start_date.isoformat(),
        "as_of_date": target.isoformat(),
        "days_worked": days_worked,
        "months_worked": months_worked,
        "annual_days": payload.annual_days,
        "accrued_days": accrued,
        "used_days": payload.used_days,
        "remaining_days": remaining,
        "eligible_for_full_leave": days_worked >= 182,
        "legal_basis": (
            "Стаття 6 Закону «Про відпустки» — тривалість; стаття 10 — право на "
            "повну відпустку після шести місяців безперервної роботи"
        ),
        "note": (
            "Пропорційний розрахунок. Пільгові категорії та додаткові відпустки "
            "не враховані."
        ),
    }


# --------------------------------------------------------------------------
# Tool 3 - submit_leave_request (write, requires confirmation)
# --------------------------------------------------------------------------


class LeaveRequestInput(BaseModel):
    """Input contract for submit_leave_request."""

    employee_id: str = Field(description="Employee identifier, format emp_NNN.")
    start_date: date = Field(description="First day of requested leave, ISO 8601.")
    days: int = Field(ge=1, le=59, description="Number of calendar days requested.")
    confirmed: bool = Field(
        default=False,
        description="Must be true to actually write the request. Leave false to "
        "get a preview the user can approve first.",
    )

    @field_validator("employee_id")
    @classmethod
    def known_format(cls, value: str) -> str:
        if not EMPLOYEE_ID_RE.match(value):
            raise ValueError("employee_id must look like emp_001")
        return value


def submit_leave_request(
    employee_id: str,
    start_date: str,
    days: int,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Record a leave request. Writes nothing until confirmed is true."""
    try:
        payload = LeaveRequestInput(
            employee_id=employee_id,
            start_date=start_date,
            days=days,
            confirmed=confirmed,
        )
    except ValidationError as error:
        raise ToolError(_format_validation_error(error)) from error

    employee = MOCK_EMPLOYEES.get(payload.employee_id)
    if employee is None:
        raise ToolError(f"employee {payload.employee_id} not found")

    entitlement = calculate_vacation_entitlement(
        employment_start_date=employee["employment_start_date"],
        as_of_date=payload.start_date.isoformat(),
    )

    preview = {
        "employee_id": payload.employee_id,
        "employee_name": employee["name"],
        "start_date": payload.start_date.isoformat(),
        "days": payload.days,
        "accrued_days_at_start": entitlement["accrued_days"],
        "exceeds_accrued": payload.days > entitlement["accrued_days"],
    }

    if not payload.confirmed:
        return {
            **preview,
            "status": "requires_confirmation",
            "written": False,
            "message": (
                "Це дія, що змінює дані. Покажи користувачу ці параметри й "
                "виклич інструмент повторно з confirmed=true лише після того, "
                "як користувач їх підтвердить."
            ),
        }

    store = (
        json.loads(LEAVE_REQUESTS_PATH.read_text(encoding="utf-8"))
        if LEAVE_REQUESTS_PATH.exists()
        else {"requests": []}
    )
    record = {
        **preview,
        "request_id": f"req_{len(store['requests']) + 1:04d}",
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "status": "submitted",
    }
    store["requests"].append(record)
    LEAVE_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEAVE_REQUESTS_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**record, "written": True}


# --------------------------------------------------------------------------
# Registry: one place that maps a name to its schema and its implementation
# --------------------------------------------------------------------------


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(piece) for piece in item["loc"]) or "input"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


UNSUPPORTED_BY_STRICT = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for the Anthropic tools parameter, strict-compatible.

    Strict tool use rejects numeric range keywords, which Pydantic emits from
    ge/le. Stripping them does not weaken the contract — it splits it in two:
    the API schema constrains the SHAPE of the arguments, and the Pydantic
    model still constrains their VALUES when the tool actually runs. The range
    moves into the description so the model is still told what is acceptable.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False

    properties = schema.get("properties", {})
    for spec in properties.values():
        spec.pop("title", None)
        bounds = [
            f"{key} {spec.pop(key)}" for key in UNSUPPORTED_BY_STRICT if key in spec
        ]
        if bounds:
            spec["description"] = (
                f"{spec.get('description', '').rstrip('.')}. "
                f"Допустимий діапазон: {', '.join(bounds)}."
            ).strip()

    schema["required"] = list(properties)
    return schema


TOOLS: dict[str, dict[str, Any]] = {
    "get_statutory_amount": {
        "kind": "read",
        "function": get_statutory_amount,
        "model": StatutoryAmountInput,
        "description": (
            "Повертає чинну на задану дату суму державного показника: мінімальна "
            "заробітна плата (місячна або погодинна), прожитковий мінімум для "
            "працездатних осіб, судовий збір у трудовому спорі. "
            "ВИКЛИКАТИ, коли користувач питає конкретну СУМУ у гривнях. "
            "НЕ ВИКЛИКАТИ, коли питання про визначення поняття, порядок "
            "нарахування чи права працівника — це є в законодавстві, "
            "використовуй search_labour_law."
        ),
    },
    "calculate_vacation_entitlement": {
        "kind": "read",
        "function": calculate_vacation_entitlement,
        "model": VacationInput,
        "description": (
            "Рахує, скільки днів щорічної відпустки накопичив конкретний "
            "працівник станом на дату, і чи має він право на повну відпустку. "
            "ВИКЛИКАТИ, коли в питанні є дата працевлаштування або йдеться про "
            "«скільки днів я вже накопичив». "
            "НЕ ВИКЛИКАТИ для питань про тривалість відпустки взагалі — "
            "це норма закону, використовуй search_labour_law."
        ),
    },
    "submit_leave_request": {
        "kind": "write",
        "function": submit_leave_request,
        "model": LeaveRequestInput,
        "description": (
            "Подає заявку на відпустку. ЦЕ ДІЯ, ЩО ЗМІНЮЄ ДАНІ. "
            "Спершу завжди виклич із confirmed=false, покажи користувачу "
            "параметри заявки й дочекайся явного підтвердження. Лише після "
            "цього виклич повторно з confirmed=true."
        ),
    },
}


def anthropic_tool_specs() -> list[dict[str, Any]]:
    """Tool definitions in the shape the Messages API expects."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "strict": True,
            "input_schema": _schema_for(spec["model"]),
        }
        for name, spec in TOOLS.items()
    ]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool. Raises ToolError on any invalid input."""
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}")
    return spec["function"](**arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one external tool directly")
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--schemas", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, spec in TOOLS.items():
            print(f"{name}  [{spec['kind']}]")
            print(f"  {spec['description'][:120]}...")
        return

    if args.schemas:
        print(json.dumps(anthropic_tool_specs(), ensure_ascii=False, indent=2))
        return

    if not args.tool:
        parser.error("provide a tool name, --list or --schemas")

    try:
        result = call_tool(args.tool, json.loads(args.arguments))
    except ToolError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
