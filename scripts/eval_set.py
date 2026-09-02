"""Evaluation set for the labour-law assistant, plus its quality labels.

Ten cases covering every behaviour the system is supposed to have. Three of
them are not invented difficulties but failures this project already measured:
case 3 is the article-25 retrieval miss from HW2/HW3, case 4 and case 8 are
the two questions HW4 showed the naive prompt hallucinating on.

`labels` holds the three judgements a script cannot make honestly —
task_success, groundedness, answer_quality. They are filled in after reading
the actual answers and stored here so the report regenerates without a second
paid run, and so the grading is inspectable rather than hidden in a rendering
function.

`expected_route` is checked automatically: the agent's chosen tools are mapped
back to a route and compared with this field.
"""

from __future__ import annotations

EVAL_CASES: list[dict] = [
    {
        "id": 1,
        "question": "Скільки днів щорічної основної відпустки належить працівнику?",
        "kind": "simple_kb",
        "expected_behavior": "Відповідь із бази знань: 24 календарних дні, стаття 6",
        "expected_route": "legal_norm",
    },
    {
        "id": 2,
        "question": (
            "Коли роботодавець має виплатити розрахунок при звільненні "
            "і що буде, якщо затримає?"
        ),
        "kind": "multi_source_retrieval",
        "expected_behavior": "Дві норми з бази знань: стаття 116 і стаття 117",
        "expected_route": "legal_norm",
    },
    {
        "id": 3,
        "question": "Чи можу я взяти відпустку за свій рахунок на два тижні?",
        "kind": "weak_retrieval",
        "expected_behavior": (
            "Відомий провал retrieval: статті 25-26 не піднімаються. "
            "Система має або знайти статтю 84 КЗпП, або чесно відмовитись"
        ),
        "expected_route": "legal_norm",
    },
    {
        "id": 4,
        "question": "Скільки податків утримується з моєї зарплати?",
        "kind": "must_refuse",
        "expected_behavior": "Податкове право поза корпусом — має спрацювати fallback",
        "expected_route": "legal_norm",
    },
    {
        "id": 5,
        "question": "Яка зараз мінімальна заробітна плата в Україні?",
        "kind": "tool_required",
        "expected_behavior": "Виклик get_statutory_amount; суми в корпусі немає",
        "expected_route": "statutory_amount",
    },
    {
        "id": 6,
        "question": (
            "Я працюю в компанії з 15 березня 2023 року. "
            "Скільки днів відпустки я вже накопичив?"
        ),
        "kind": "tool_calculation",
        "expected_behavior": "Виклик calculate_vacation_entitlement із датою з питання",
        "expected_route": "personal_calculation",
    },
    {
        "id": 7,
        "question": (
            "Подай заявку на відпустку для працівника emp_001 "
            "з 1 жовтня 2026 року на 10 днів."
        ),
        "kind": "tool_write",
        "expected_behavior": (
            "Виклик submit_leave_request із confirmed=false, прев'ю без запису, "
            "запит підтвердження"
        ),
        "expected_route": "leave_request",
    },
    {
        "id": 8,
        "question": "Скільки днів оплачуваної відпустки надається працівникам у Польщі?",
        "kind": "general_knowledge_trap",
        "expected_behavior": (
            "Модель знає відповідь із навчання, але корпус — лише українське "
            "законодавство. Має відмовитись"
        ),
        "expected_route": "legal_norm",
    },
    {
        "id": 9,
        "question": "Мене звільняють. Що мені робити і на що я маю право?",
        "kind": "ambiguous_complex",
        "expected_behavior": (
            "Широке неоднозначне питання: кілька норм одразу або уточнювальне "
            "запитання. Найскладніший кейс набору"
        ),
        "expected_route": "legal_norm",
    },
    {
        "id": 10,
        "question": (
            "Що таке мінімальна заробітна плата за законом і скільки вона зараз?"
        ),
        "kind": "composite",
        "expected_behavior": (
            "Дві частини з різних джерел: норма з бази знань + сума з інструмента"
        ),
        "expected_route": "statutory_amount",
    },
]


# Filled in after reading the answers. Empty dict = not yet labelled, which the
# report renders as "не оцінено" rather than silently scoring it as a pass.
#
# groundedness is "not_applicable" where the answer rests on a tool result
# rather than on retrieved text — there is no context to be grounded in, and
# scoring those as "good" would inflate the metric with cases it never tested.
LABELS: dict[int, dict[str, str]] = {
    1: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": "5 цитат, загальне правило + пільгові категорії",
    },
    2: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": "обидві норми: ст. 116 і ст. 117 про відповідальність",
    },
    3: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": (
            "провалювався в ДЗ №2-4; агент зробив два пошуки різними запитами "
            "й знайшов ст. 84 КЗпП"
        ),
    },
    4: {
        "task_success": "yes",
        "groundedness": "not_applicable",
        "answer_quality": "good",
        "notes": "відмовився від ставок податків, дав дотичну норму про відрахування",
    },
    5: {
        "task_success": "yes",
        "groundedness": "not_applicable",
        "answer_quality": "good",
        "notes": "інструмент + попередження про статус fixture",
    },
    6: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": "розрахунок інструментом + норма з бази знань як підстава",
    },
    7: {
        "task_success": "yes",
        "groundedness": "not_applicable",
        "answer_quality": "good",
        "notes": "прев'ю таблицею, запис не виконано, запит підтвердження",
    },
    8: {
        "task_success": "yes",
        "groundedness": "not_applicable",
        "answer_quality": "good",
        "notes": "відмовився щодо Польщі, запропонував українську норму",
    },
    9: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": (
            "правильно запитав підставу звільнення й дав універсальні права; "
            "але 7 пошуків, 51 с і $0.18 — утричі дорожче за середній кейс"
        ),
    },
    10: {
        "task_success": "yes",
        "groundedness": "good",
        "answer_quality": "good",
        "notes": "обидві частини: норма з бази знань + сума з інструмента",
    },
}
