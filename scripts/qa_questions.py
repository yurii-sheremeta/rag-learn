"""The test question set for the grounded QA pipeline.

Ten questions covering the four behaviours the assignment asks for, plus the
two failure modes this project already measured in HW2 and HW3.

`kind` drives nothing in the pipeline — it exists so the report can group
questions by what they are meant to probe, and so a fallback that fires on a
`grounded` question is visibly a bug rather than a judgement call.

    grounded      the answer is in the corpus and retrieval finds it
    reformulated  same norm, conversational wording
    multi_source  needs two articles to answer completely
    weak_context  retrieval is known to return the wrong chunks confidently
    out_of_scope  plausible labour question, but no act in the corpus covers it
    general_trap  the model knows the answer from training, corpus does not

`context_sufficient` says whether the retrieved chunks actually contain the
answer. It is not an assumption: every value was checked against the cached
context. It is deliberately separate from `kind`, because a question can be a
fair reformulation (a property of the question) and still arrive with useless
context (a property of retrieval) — q03 turned out to be exactly that, and
conflating the two would have scored a correct refusal as a failure.
"""

from __future__ import annotations

TEST_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q01_vacation_days",
        "context_sufficient": True,
        "kind": "grounded",
        "question": "Скільки днів щорічної основної відпустки мені належить?",
        "expected": "24 календарних дні (стаття 6 Закону «Про відпустки»)",
    },
    {
        "id": "q02_working_hours",
        "context_sufficient": True,
        "kind": "grounded",
        "question": "Яка максимальна тривалість робочого тижня?",
        "expected": "40 годин (стаття 50 КЗпП)",
    },
    {
        "id": "q03_vacation_reformulated",
        "context_sufficient": False,
        "kind": "reformulated",
        "question": (
            "Я щойно влаштувався на роботу. Через скільки часу зможу піти "
            "у відпустку?"
        ),
        "expected": (
            "норма про шість місяців безперервної роботи міститься у "
            "zakon_pro_vidpustky_chunk_0021, але retrieval повернув чанки "
            "0006, 0026 і 0007 — жоден її не містить. Перевірено; правильна "
            "поведінка тут — відмова"
        ),
    },
    {
        "id": "q04_settlement_and_penalty",
        "context_sufficient": True,
        "kind": "multi_source",
        "question": (
            "Коли роботодавець має виплатити розрахунок при звільненні "
            "і що буде, якщо він затримає виплату?"
        ),
        "expected": "стаття 116 (день звільнення) + стаття 117 (відповідальність за затримку)",
    },
    {
        "id": "q05_dismissal_sick_leave_martial_law",
        "context_sufficient": True,
        "kind": "multi_source",
        "question": (
            "Чи можуть мене звільнити під час воєнного стану, поки я на лікарняному?"
        ),
        "expected": (
            "стаття 5 Закону про воєнний стан дозволяє, на відміну від "
            "загального правила статті 40 КЗпП"
        ),
    },
    {
        "id": "q06_unpaid_leave",
        "context_sufficient": False,
        "kind": "weak_context",
        "question": "Чи можу я взяти відпустку за свій рахунок на два тижні?",
        "expected": (
            "статті 25-26 Закону «Про відпустки» / стаття 84 КЗпП — але retrieval "
            "у HW2 і HW3 стабільно не піднімав їх у top-3"
        ),
    },
    {
        "id": "q07_moonlighting",
        "context_sufficient": False,
        "kind": "weak_context",
        "question": "Чи можу я офіційно працювати на двох роботах одночасно?",
        "expected": "сумісництво згадане в КЗпП побіжно; повної відповіді корпус не містить",
    },
    {
        "id": "q08_income_tax",
        "context_sufficient": False,
        "kind": "out_of_scope",
        "question": "Скільки податків утримується з моєї зарплати?",
        "expected": "податкове право, у корпусі з шести актів відсутнє — має спрацювати fallback",
    },
    {
        "id": "q09_poland_vacation",
        "context_sufficient": False,
        "kind": "general_trap",
        "question": "Скільки днів оплачуваної відпустки надається працівникам у Польщі?",
        "expected": (
            "модель знає відповідь із загальних знань, але корпус — виключно "
            "українське законодавство; має відмовитись"
        ),
    },
    {
        "id": "q10_court_claim",
        "context_sufficient": False,
        "kind": "out_of_scope",
        "question": "Яке мито треба сплатити, щоб подати позов на роботодавця до суду?",
        "expected": (
            "КЗпП описує трудові спори, але розмір судового збору — інший закон; "
            "часткового контексту недостатньо"
        ),
    },
]
