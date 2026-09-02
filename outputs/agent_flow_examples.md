# Трасування agentic workflow

Детермінований rule-based роутер — жодного виклику LLM, тому прогін безкоштовний і повторюється байт у байт.

Згенеровано: 2026-09-02 · відтворюється командою `python scripts/run_agent_flow.py`

---

## 1. Яка зараз мінімальна заробітна плата?

*Перевіряє: проста маршрутизація на довідник сум*

```
Question: Яка зараз мінімальна заробітна плата?
Route: statutory_amount
  reason: збіг за ключем «мінімальна заробітна плата» → Конкретна сума у гривнях із державного довідника
Plan: extract_slots -> lookup_amount -> compose_answer

Step 1: extract_slots
  Observation: {"indicator": "minimum_wage_monthly"}

Step 2: lookup_amount
  Tool called: get_statutory_amount
  Input: {"indicator": "minimum_wage_monthly", "on_date": null}
  Observation: {"indicator": "minimum_wage_monthly", "label": "Мінімальна заробітна плата
               (місячна)", "value": 8647, "unit": "UAH/month", "effective_from": "2026-01-01",
               "effective_to": null, "on_date": "2026-09-02", "legal_basis": "Стаття 3 Закону
               України «Про оплату праці»; розмір встановлюється Законом про Держбюджет",

State after run:
  {
    "user_goal": "Яка зараз мінімальна заробітна плата?",
    "selected_route": "statutory_amount",
    "slots": {
      "indicator": "minimum_wage_monthly"
    },
    "tool_calls": [
      "get_statutory_amount"
    ],
    "steps_executed": [
      "extract_slots",
      "lookup_amount"
    ],
    "needs_user_input": null
  }

Final answer:
Мінімальна заробітна плата (місячна) станом на 2026-09-02 — 8647 UAH/month.
Діє з 2026-01-01. Підстава: Стаття 3 Закону України «Про оплату праці»; розмір
встановлюється Законом про Держбюджет.
[get_statutory_amount]
Увага: значення демонстраційне і не звірене з чинним Законом про Держбюджет.
```

**Очікувався маршрут:** `statutory_amount`

---

## 2. Скільки днів щорічної відпустки належить за законом?

*Перевіряє: норма права — має піти в базу знань, а не в інструмент*

```
Question: Скільки днів щорічної відпустки належить за законом?
Route: legal_norm
  reason: збіг за ключем «закон» → Норма права — відповідає база знань
Plan: extract_slots -> search_knowledge_base -> compose_answer

Step 1: extract_slots
  Observation: {}

Step 2: search_knowledge_base
  Tool called: search_knowledge_base
  Input: {"query": "Скільки днів щорічної відпустки належить за законом?"}
  Observation: {"source": "faiss_index", "chunks": [{"chunk_id": "zakon_pro_vidpustky_chunk_0055",
               "score": 0.879, "section": "Стаття 19-1. Відпустка при народженні дитини",
               "source_file": "data/raw/zakon_pro_vidpustky.html", "text": "для державної
               реєстрації народження дитини, або свідоцтва про народження дитини протягом 30

State after run:
  {
    "user_goal": "Скільки днів щорічної відпустки належить за законом?",
    "selected_route": "legal_norm",
    "slots": {},
    "tool_calls": [
      "search_knowledge_base"
    ],
    "steps_executed": [
      "extract_slots",
      "search_knowledge_base"
    ],
    "needs_user_input": null
  }

Final answer:
для державної реєстрації народження дитини, або свідоцтва про народження дитини
протягом 30 календарних днів з дня виходу в таку відпустку. Порядок надання відпустки
при народженні дитини встановлюється Кабінетом Міністрів України. Тривалість відпустки
у зв'язку з вагітністю та пологами обчислюється сумарно і становить 126 календарних
днів (140 календарних днів - у разі народження двох і більше ді

[Стаття 19-1. Відпустка при народженні дитини, zakon_pro_vidpustky_chunk_0055]
Джерело: data/raw/zakon_pro_vidpustky.html
```

**Очікувався маршрут:** `legal_norm`

---

## 3. Я працюю з 15 березня 2023 року. Скільки днів відпустки я вже накопичив?

*Перевіряє: витяг дати з тексту словами + розрахунок*

```
Question: Я працюю з 15 березня 2023 року. Скільки днів відпустки я вже накопичив?
Route: personal_calculation
  reason: збіг за ключем «накопич» → Розрахунок за датами конкретного працівника
Plan: extract_slots -> check_required_slots -> calculate_entitlement -> compose_answer

Step 1: extract_slots
  Observation: {"dates": ["2023-03-15"]}

Step 2: check_required_slots
  Observation: {"employment_start_date": "2023-03-15"}

Step 3: calculate_entitlement
  Tool called: calculate_vacation_entitlement
  Input: {"employment_start_date": "2023-03-15"}
  Observation: {"employment_start_date": "2023-03-15", "as_of_date": "2026-09-02", "days_worked":
               1267, "months_worked": 42, "annual_days": 24, "accrued_days": 24, "used_days": 0,
               "remaining_days": 24, "eligible_for_full_leave": true, "legal_basis": "Стаття 6
               Закону «Про відпустки» — тривалість; стаття 10 — право на повну відпустку після

State after run:
  {
    "user_goal": "Я працюю з 15 березня 2023 року. Скільки днів відпустки я вже накопичив?",
    "selected_route": "personal_calculation",
    "slots": {
      "dates": [
        "2023-03-15"
      ]
    },
    "tool_calls": [
      "calculate_vacation_entitlement"
    ],
    "steps_executed": [
      "extract_slots",
      "check_required_slots",
      "calculate_entitlement"
    ],
    "needs_user_input": null
  }

Final answer:
Станом на 2026-09-02 ви відпрацювали 1267 днів (близько 42 місяців).
Накопичено 24 з 24 днів щорічної відпустки, залишок 24.
Право на повну відпустку вже настало.
[calculate_vacation_entitlement] Підстава: Стаття 6 Закону «Про відпустки» —
тривалість; стаття 10 — право на повну відпустку після шести місяців безперервної
роботи
```

**Очікувався маршрут:** `personal_calculation`

---

## 4. Подай заявку на відпустку для emp_001 з 01.10.2026 на 10 днів

*Перевіряє: багатокроковий маршрут: слоти → розрахунок → прев'ю → підтвердження*

```
Question: Подай заявку на відпустку для emp_001 з 01.10.2026 на 10 днів
Route: leave_request
  reason: збіг за ключем «подай заявку» → Подання заявки на відпустку — дія, що змінює дані
Plan: extract_slots -> check_required_slots -> calculate_entitlement -> preview_request -> compose_answer

Step 1: extract_slots
  Observation: {"employee_id": "emp_001", "dates": ["2026-10-01"], "days": 10}

Step 2: check_required_slots
  Observation: {"employee_id": "emp_001", "start_date": "2026-10-01", "days": 10}

Step 3: calculate_entitlement
  Tool called: calculate_vacation_entitlement
  Input: {"employment_start_date": "2023-03-15", "as_of_date": "2026-10-01"}
  Observation: {"employment_start_date": "2023-03-15", "as_of_date": "2026-10-01", "days_worked":
               1296, "months_worked": 43, "annual_days": 24, "accrued_days": 24, "used_days": 0,
               "remaining_days": 24, "eligible_for_full_leave": true, "legal_basis": "Стаття 6
               Закону «Про відпустки» — тривалість; стаття 10 — право на повну відпустку після

Step 4: preview_request
  Tool called: submit_leave_request
  Input: {"employee_id": "emp_001", "start_date": "2026-10-01", "days": 10, "confirmed": false}
  Observation: {"employee_id": "emp_001", "employee_name": "Олена Ткаченко", "start_date":
               "2026-10-01", "days": 10, "accrued_days_at_start": 24, "exceeds_accrued": false,
               "status": "requires_confirmation", "written": false, "message": "Це дія, що змінює
               дані. Покажи користувачу ці параметри й виклич інструмент повторно з confirmed=true

State after run:
  {
    "user_goal": "Подай заявку на відпустку для emp_001 з 01.10.2026 на 10 днів",
    "selected_route": "leave_request",
    "slots": {
      "employee_id": "emp_001",
      "dates": [
        "2026-10-01"
      ],
      "days": 10
    },
    "tool_calls": [
      "calculate_vacation_entitlement",
      "submit_leave_request"
    ],
    "steps_executed": [
      "extract_slots",
      "check_required_slots",
      "calculate_entitlement",
      "preview_request"
    ],
    "needs_user_input": "Підтвердіть подання заявки."
  }

Final answer:
Готую заявку: Олена Ткаченко (emp_001), з 2026-10-01 на 10 днів. Накопичено на цю
дату: 24 днів.
Заявку ще НЕ подано — підтвердіть, і я її надішлю.
[submit_leave_request, status=requires_confirmation]
```

**Очікувався маршрут:** `leave_request`

---

## 5. Хочу подати заявку на відпустку

*Перевіряє: перехід маршруту: слотів бракує, workflow питає користувача*

```
Question: Хочу подати заявку на відпустку
Route: leave_request
  reason: збіг за ключем «подати заявку» → Подання заявки на відпустку — дія, що змінює дані
Plan: extract_slots -> check_required_slots -> calculate_entitlement -> preview_request -> compose_answer -> ask_user

Step 1: extract_slots
  Observation: {}

Step 2: check_required_slots   [слотів бракує]
  Observation: {"missing": "ідентифікатор працівника у форматі emp_001"}

State after run:
  {
    "user_goal": "Хочу подати заявку на відпустку",
    "selected_route": "leave_request",
    "slots": {},
    "tool_calls": [],
    "steps_executed": [
      "extract_slots",
      "check_required_slots"
    ],
    "needs_user_input": "Щоб продовжити, вкажіть, будь ласка, ідентифікатор працівника у форматі emp_001."
  }

Final answer:
Щоб продовжити, вкажіть, будь ласка, ідентифікатор працівника у форматі emp_001.
```

**Очікувався маршрут:** `leave_request → ask_user`

---

## 6. Розкажи щось цікаве

*Перевіряє: нічого не збіглося — fallback*

```
Question: Розкажи щось цікаве
Route: clarification
  reason: жодне правило не спрацювало
Plan: compose_clarification

Step 1: compose_clarification
  Observation: "Уточніть, будь ласка: вас цікавить норма закону, конкретна сума (мінімальна
               зарплата, прожитковий мінімум), розрахунок вашої відпустки чи подання заявки?"

State after run:
  {
    "user_goal": "Розкажи щось цікаве",
    "selected_route": "clarification",
    "slots": {},
    "tool_calls": [],
    "steps_executed": [
      "compose_clarification"
    ],
    "needs_user_input": "Уточніть, будь ласка: вас цікавить норма закону, конкретна сума (мінімальна зарплата, прожитковий мінімум), розрахунок вашої відпустки чи подання заявки?"
  }

Final answer:
Уточніть, будь ласка: вас цікавить норма закону, конкретна сума (мінімальна зарплата,
прожитковий мінімум), розрахунок вашої відпустки чи подання заявки?
```

**Очікувався маршрут:** `clarification`

---
