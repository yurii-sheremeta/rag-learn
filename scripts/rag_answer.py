"""Grounded QA: question -> retrieval -> prompt -> Claude -> cited answer.

The retrieval half of this pipeline is HW2/HW3; this module adds the last
link. Its whole job is to make the model answer from the six retrieved acts
and nothing else — including refusing when the retrieved context does not
actually contain the answer.

Why the refusal cannot be a score threshold: HW2 measured every top-1 score
inside the narrow band 0.851-0.897, with correct and incorrect answers often
0.01 apart. There is no cutoff that separates them, so judging sufficiency is
delegated to the model, which sees the text rather than a number.

Usage:
    python scripts/rag_answer.py "скільки днів відпустки мені належить"
    python scripts/rag_answer.py --question-id q06_unpaid_leave --prompt v1
    python scripts/rag_answer.py "..." --json

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "outputs" / "retrieval_cache.json"

MODEL = "claude-opus-5"
MAX_TOKENS = 4000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

INSUFFICIENT_MARKER = "недостатньо інформації"


@dataclass
class Answer:
    question: str
    prompt_version: str
    text: str
    chunks: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        """True when the model declined for lack of context."""
        return INSUFFICIENT_MARKER in self.text.lower()

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c["chunk_id"] for c in self.chunks if c["chunk_id"] in self.text]

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            source = chunk["metadata"]["source_file"]
            if source not in seen:
                seen.append(source)
        return seen

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * 5.0 + self.output_tokens / 1e6 * 25.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "prompt_version": self.prompt_version,
            "answer": self.text,
            "refused": self.refused,
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "section": c["metadata"]["section"],
                    "source_file": c["metadata"]["source_file"],
                }
                for c in self.chunks
            ],
            "cited_chunk_ids": self.cited_chunk_ids,
            "sources": self.sources,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 5),
            },
            "stop_reason": self.stop_reason,
            "notes": self.notes,
        }


def load_cached_chunks(question_id: str) -> list[dict]:
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"{CACHE_PATH} not found — run scripts/build_retrieval_cache.py first"
        )
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    entry = cache["questions"].get(question_id)
    if entry is None:
        raise SystemExit(f"question {question_id!r} is not in the cache")
    return entry["chunks"]


def retrieve_live(question: str, k: int, use_rerank: bool) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from retrieval_improved import ImprovedRetriever

    retriever = ImprovedRetriever(load_reranker=use_rerank)
    hits = retriever.search(question, k=k, use_rerank=use_rerank)
    return [
        {
            "chunk_id": hit.chunk_id,
            "score": round(hit.score, 4),
            "text": hit.text,
            "metadata": hit.metadata,
        }
        for hit in hits
    ]


def ask_claude(system: str, user: str) -> tuple[str, dict]:
    """Send one grounded-QA turn to Claude and return (text, usage)."""
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it in your shell and rerun; "
            "the key is never read from or written to this repository."
        )

    client = anthropic.Anthropic()
    request = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": user}],
    }

    notes: list[str] = []
    try:
        response = client.beta.messages.create(
            **request, betas=[FALLBACK_BETA], fallbacks="default"
        )
    except Exception as error:
        notes.append(f"server-side fallbacks unavailable ({type(error).__name__})")
        response = client.messages.create(**request)

    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
        "notes": notes,
    }
    return text, usage


def answer(
    question: str,
    chunks: list[dict],
    version: str = "v3",
) -> Answer:
    from prompts import render

    system, user = render(version, question, chunks)
    text, usage = ask_claude(system, user)

    return Answer(
        question=question,
        prompt_version=version,
        text=text,
        chunks=chunks,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        stop_reason=usage["stop_reason"],
        notes=usage["notes"],
    )


def print_answer(result: Answer) -> None:
    print(f"\nQuestion: {result.question}")
    print("=" * 78)
    print("\nRetrieved chunks:")
    for chunk in result.chunks:
        marker = "cited" if chunk["chunk_id"] in result.text else "     "
        print(
            f"  [{marker}] {chunk['chunk_id']:34s} score {chunk['score']:.4f}  "
            f"{chunk['metadata']['section'][:44]}"
        )

    print("\nAnswer:")
    for paragraph in result.text.split("\n"):
        if not paragraph.strip():
            print()
            continue
        for line in textwrap.wrap(paragraph, width=76):
            print(f"  {line}")

    print(f"\nSource: {', '.join(result.sources)}")
    print(
        f"Prompt: {result.prompt_version} | "
        f"tokens {result.input_tokens}+{result.output_tokens} | "
        f"${result.cost_usd:.4f}"
        + (" | fallback triggered" if result.refused else "")
    )
    for note in result.notes:
        print(f"note: {note}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded QA over the labour law corpus")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--question-id", help="use a cached test question by id")
    parser.add_argument("--prompt", default="v3", help="prompt version: v1, v2 or v3")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.question_id:
        from qa_questions import TEST_QUESTIONS

        known = {q["id"]: q for q in TEST_QUESTIONS}
        if args.question_id not in known:
            raise SystemExit(f"unknown question id {args.question_id!r}")
        question = known[args.question_id]["question"]
        chunks = load_cached_chunks(args.question_id)
    elif args.question:
        question = " ".join(args.question)
        chunks = retrieve_live(question, k=args.k, use_rerank=not args.no_rerank)
    else:
        parser.error("provide a question or --question-id")

    result = answer(question, chunks, version=args.prompt)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_answer(result)


if __name__ == "__main__":
    sys.exit(main())
