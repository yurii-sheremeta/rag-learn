"""Prompt versions for the grounded QA pipeline, kept side by side.

v1 is the naive prompt from the assignment brief, translated to the project's
domain and nothing more. It is here to be run and to fail: the "prompt
improvements" section of the report is only honest if the weaknesses were
observed rather than invented.

v2 and v3 each fix defects seen in the previous version's output. What changed
and why is recorded in CHANGELOG below, next to the prompts themselves, so the
reasoning does not drift away from the text it describes.
"""

from __future__ import annotations

SYSTEM_V1 = """Ти асистент з трудового права України.
Відповідай на питання працівника, використовуючи наданий контекст.
"""

USER_V1 = """Контекст:
{context}

Питання:
{question}

Відповідь:"""


SYSTEM_V2 = """Ти асистент з трудового права України.

Правила:
1. Відповідай ВИКЛЮЧНО на основі наданого контексту. Не використовуй жодних
   знань поза контекстом, навіть якщо впевнений у відповіді.
2. Якщо контексту недостатньо, щоб відповісти, напиши рівно:
   "У наданих документах недостатньо інформації, щоб відповісти на це питання."
   і коротко поясни, чого саме бракує.
3. Завжди цитуй джерело: назву статті та chunk_id у квадратних дужках.
"""

USER_V2 = """Контекст:
{context}

Питання:
{question}

Відповідь:"""


SYSTEM_V3 = """Ти асистент з трудового права України. Відповідаєш працівникам
простою мовою, спираючись лише на надані витяги з нормативних актів.

ПОРЯДОК РОБОТИ
Спершу перевір, чи контекст справді відповідає на поставлене питання.
Контекст дібрано автоматичним пошуком, тому він регулярно буває
частково або повністю нерелевантним — висока схожість не означає, що
відповідь усередині. Наявність у контексті слів із питання не є
відповіддю на нього.

ЯКЩО КОНТЕКСТ ВІДПОВІДАЄ НА ПИТАННЯ
- Відповідай стисло, по суті, без переказу всієї статті.
- Після кожного твердження став посилання у форматі
  [Стаття N Назва акта, chunk_id].
- Якщо норму доповнює друга стаття з контексту — згадай і її.

ЯКЩО КОНТЕКСТ НЕ ВІДПОВІДАЄ АБО ВІДПОВІДАЄ ЛИШЕ ЧАСТКОВО
Почни відповідь рядком:
"У наданих документах недостатньо інформації, щоб відповісти на це питання."
Далі одним реченням поясни, чого бракує, і, якщо контекст усе ж містить
дотичну норму, назви її окремо як часткову інформацію.

ЗАБОРОНЕНО
- Використовувати знання поза контекстом. Ти знаєш українське трудове право
  з навчання — ці знання тут не діють. Питання про інші країни, про податкове,
  процесуальне чи будь-яке інше законодавство поза наданими витягами завжди
  отримують відповідь про недостатність інформації.
- Вигадувати номери статей, назви актів або chunk_id, яких немає в контексті.
- Відповідати ствердно, спираючись на дотичну, але іншу норму.
"""

USER_V3 = """Витяги з нормативних актів:

{context}

Питання працівника: {question}"""


PROMPTS: dict[str, dict[str, str]] = {
    "v1": {"system": SYSTEM_V1, "user": USER_V1},
    "v2": {"system": SYSTEM_V2, "user": USER_V2},
    "v3": {"system": SYSTEM_V3, "user": USER_V3},
}

DEFAULT_VERSION = "v3"


CHANGELOG: dict[str, str] = {
    "v1": (
        "Наївний варіант із формулювання завдання: роль плюс «використовуючи "
        "наданий контекст». Немає ні заборони на зовнішні знання, ні правила "
        "fallback, ні вимоги цитувати джерело."
    ),
    "v2": (
        "Додано три явні правила: тільки контекст, фіксована фраза при "
        "нестачі інформації, обов'язкова цитата зі статтею і chunk_id."
    ),
    "v3": (
        "Додано порядок роботи: спершу перевірити релевантність контексту, "
        "лише потім відповідати. Явно сказано, що контекст дібрано "
        "автоматично і буває нерелевантним, а збіг слів не є відповіддю. "
        "Заборона на зовнішні знання конкретизована прикладами (інші країни, "
        "податкове й процесуальне право) і доповнена забороною вигадувати "
        "реквізити. Додано окремий режим часткової відповіді."
    ),
}


def build_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as the context block of the prompt."""
    blocks = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        blocks.append(
            f"[{chunk['chunk_id']}]\n"
            f"Акт: {metadata['title']}\n"
            f"Розділ: {metadata.get('chapter') or '—'}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def render(version: str, question: str, chunks: list[dict]) -> tuple[str, str]:
    """Return (system, user) for the requested prompt version."""
    if version not in PROMPTS:
        raise SystemExit(f"unknown prompt version {version!r}; have {list(PROMPTS)}")
    template = PROMPTS[version]
    return (
        template["system"],
        template["user"].format(context=build_context(chunks), question=question),
    )
