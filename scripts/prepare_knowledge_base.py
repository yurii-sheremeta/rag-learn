"""Build the knowledge base for the "Ukrainian labour law" RAG system.

Pipeline:

    data/raw/*.html                 raw HTML pages from zakon.rada.gov.ua
        -> normalization            clean text + reconstructed act structure
    data/interim/*.md               normalized documents (for eyeball checks)
        -> chunking                 structure-aware splitting along articles
    data/processed/chunks.jsonl     chunks with metadata (1 line = 1 chunk)

The script has no third-party dependencies — standard library only.

Usage:  python scripts/prepare_knowledge_base.py

Note on languages: comments and console output are in English, while every
literal that ends up inside a document or a chunk (structure keywords,
abbreviations, breadcrumb wording) stays in Ukrainian by design.
"""

from __future__ import annotations

import html as html_module
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Chunking parameters
# --------------------------------------------------------------------------

TARGET_CHARS = 800   # preferred chunk size
MAX_CHARS = 1000     # hard upper bound (assignment requirement)
MIN_CHARS = 500      # below this we try to append the next block instead
OVERLAP_CHARS = 150  # overlap between chunks that split a single article

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

LANGUAGE = "uk"
DOMAIN = "labour_law"
JURISDICTION = "UA"


# --------------------------------------------------------------------------
# 1. Normalization: HTML -> clean text paragraphs
# --------------------------------------------------------------------------

ARTICLE_ANCHOR = '<div id="article">'

BLOCK_CLOSE_RE = re.compile(r"</(p|div|tr|td|th|li|h[1-6]|table)\s*>", re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.I | re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
TAG_RE = re.compile(r"<[^>]+>")

# Editorial notes such as {Із змінами, внесеними згідно із Законом ...} are
# portal boilerplate that hurts retrieval quality, so we strip them.
EDITORIAL_INLINE_RE = re.compile(r"\{[^{}]*\}", re.S)

# Technical stamp of the document card on the portal.
STAMP_MARKERS = (
    "поточна редакція",
    "Документ, актуальний на",
)

# Chronological list of amending acts: "№ 2048-08 від 18.09.73, ВВР 1973,
# № 40, ст.343". Carries no legal content — pure noise for retrieval.
AMENDMENT_REF_RE = re.compile(r"^\(?№\s*\d[\w/–-]*\s+від\s+\d")
VVR_REF_RE = re.compile(r"^\(?ВВР[,\s]")


def extract_article_html(raw_html: str) -> str:
    """Cut the <div id="article"> container holding the act text out of the page."""
    start = raw_html.find(ARTICLE_ANCHOR)
    if start == -1:
        raise ValueError("article container not found — portal markup has changed")

    cursor = start + len(ARTICLE_ANCHOR)
    depth = 1
    tag_re = re.compile(r"<(/?)div\b[^>]*>", re.I)

    while depth > 0:
        match = tag_re.search(raw_html, cursor)
        if match is None:  # container never closed — take the rest of the page
            return raw_html[start:]
        depth += -1 if match.group(1) else 1
        cursor = match.end()

    return raw_html[start:cursor]


def is_noise(paragraph: str) -> bool:
    """True for paragraphs that carry no normative content."""
    if paragraph.startswith("{") and paragraph.endswith("}"):
        return True
    if any(marker in paragraph for marker in STAMP_MARKERS):
        return True
    if AMENDMENT_REF_RE.match(paragraph) or VVR_REF_RE.match(paragraph):
        return True
    # Orphaned fragments of editorial notes such as "від 01.07.2022}"
    if paragraph.endswith("}") and "{" not in paragraph:
        return True
    if paragraph.startswith("{") and "}" not in paragraph:
        return True
    return False


def html_to_paragraphs(article_html: str) -> list[str]:
    """Turn the article container's HTML into a list of clean text paragraphs."""
    text = SCRIPT_RE.sub(" ", article_html)
    text = COMMENT_RE.sub(" ", text)
    text = BR_RE.sub("\n", text)
    text = BLOCK_CLOSE_RE.sub("\n\n", text)
    text = TAG_RE.sub("", text)
    text = html_module.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")

    paragraphs: list[str] = []
    inside_note = False  # an editorial note whose braces span several paragraphs

    for raw_paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"[ \t]+", " ", raw_paragraph.replace("\n", " ")).strip()
        if not paragraph:
            continue

        opens = paragraph.count("{")
        closes = paragraph.count("}")

        if inside_note:
            # drop everything inside the note until we see it close
            if closes > opens:
                inside_note = False
            continue
        if opens > closes:
            inside_note = True
            continue

        if is_noise(paragraph):
            continue

        paragraph = EDITORIAL_INLINE_RE.sub("", paragraph)
        paragraph = re.sub(r"^\s*/-\s*", "", paragraph)  # one-off markup artefact
        paragraph = re.sub(r"\s{2,}", " ", paragraph).strip()
        if len(paragraph) < 3:
            continue
        paragraphs.append(paragraph)

    return paragraphs


# --------------------------------------------------------------------------
# 2. Reconstructing the structure of the act
# --------------------------------------------------------------------------

CHAPTER_RE = re.compile(
    r"^(Розділ|РОЗДІЛ|Глава|ГЛАВА)\s+([IVXLC]+(?:-[А-ЯҐЄІЇ])?|\d+)\.?$"
)
CHAPTER_INLINE_RE = re.compile(
    r"^(Розділ|РОЗДІЛ|Глава|ГЛАВА)\s+([IVXLC]+(?:-[А-ЯҐЄІЇ])?|\d+)[.\s]+(\S.*)$"
)
ARTICLE_RE = re.compile(
    r"^Стаття\s+(\d+(?:-\d+)?(?:-[А-ЯҐЄІЇ])?)\s*[.–-]\s*(.+)$"
)
ARTICLE_BARE_RE = re.compile(r"^Стаття\s+(\d+(?:-\d+)?(?:-[А-ЯҐЄІЇ])?)\s*\.?$")


@dataclass
class Block:
    """Smallest unit of meaning — one paragraph within its structural context."""

    chapter: str | None
    article_no: str | None
    section: str
    text: str


@dataclass
class Document:
    document_id: str
    title: str
    short_title: str
    document_type: str
    source_file: str
    source_url: str
    retrieved_at: str
    blocks: list[Block] = field(default_factory=list)


def parse_structure(paragraphs: list[str]) -> list[Block]:
    """Attach the chapter and the article each paragraph belongs to."""
    blocks: list[Block] = []
    chapter: str | None = None
    article_no: str | None = None
    section = "Преамбула"
    pending_chapter_number: str | None = None

    for paragraph in paragraphs:
        # The chapter name usually sits in its own paragraph right after the number.
        if pending_chapter_number is not None:
            chapter = f"{pending_chapter_number}. {paragraph.strip()}"
            pending_chapter_number = None
            continue

        if CHAPTER_RE.match(paragraph):
            pending_chapter_number = paragraph.rstrip(".")
            continue

        inline_chapter = CHAPTER_INLINE_RE.match(paragraph)
        if inline_chapter:
            kind, number, name = inline_chapter.groups()
            chapter = f"{kind} {number}. {name.strip()}"
            continue

        article = ARTICLE_RE.match(paragraph)
        if article:
            article_no, name = article.groups()
            section = f"Стаття {article_no}. {name.strip()}"
            continue

        bare_article = ARTICLE_BARE_RE.match(paragraph)
        if bare_article:
            article_no = bare_article.group(1)
            section = f"Стаття {article_no}"
            continue

        blocks.append(
            Block(
                chapter=chapter,
                article_no=article_no,
                section=section,
                text=paragraph,
            )
        )

    return blocks


def write_normalized_markdown(document: Document) -> Path:
    """Store the normalized document as Markdown — an intermediate artefact."""
    lines = [f"# {document.title}", "", f"Джерело: {document.source_url}", ""]
    current_chapter: str | None = None
    current_section: str | None = None

    for block in document.blocks:
        if block.chapter != current_chapter:
            current_chapter = block.chapter
            current_section = None
            if current_chapter:
                lines += ["", f"## {current_chapter}", ""]
        if block.section != current_section:
            current_section = block.section
            lines += ["", f"### {current_section}", ""]
        lines.append(block.text)
        lines.append("")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    target = INTERIM_DIR / f"{document.document_id}.md"
    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# 3. Chunking
# --------------------------------------------------------------------------

# Ukrainian legal abbreviations after which a period does not end a sentence.
ABBREVIATIONS = {
    "ст", "стст", "п", "пп", "ч", "чч", "абз", "розд", "гл", "ін", "т", "д",
    "грн", "тис", "млн", "млрд", "год", "хв", "коп", "р", "рр", "м", "см",
    "кв", "проц", "напр", "зокр", "ім", "обл", "буд", "вул",
}

SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЇІЄҐA-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, honouring Ukrainian legal abbreviations."""
    sentences: list[str] = []
    start = 0

    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        head = text[start:match.start()]
        last_token = re.search(r"([^\s.]+)\.$", head.strip())
        if last_token:
            token = last_token.group(1).lower()
            # an abbreviation like "ст.", "п.", "ч." or a short item number
            if token in ABBREVIATIONS or (token.isdigit() and len(token) <= 2):
                continue
        sentences.append(head.strip())
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]


def hard_split(sentence: str, limit: int) -> list[str]:
    """Last-resort split of an over-long sentence: by commas first, then by words."""
    if len(sentence) <= limit:
        return [sentence]

    pieces: list[str] = []
    buffer = ""
    for part in re.split(r"(?<=[,;:])\s+", sentence):
        candidate = f"{buffer} {part}".strip()
        if len(candidate) <= limit or not buffer:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = part
    if buffer:
        pieces.append(buffer)

    result: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            result.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            result.append(piece)
    return result


def build_header(document: Document, block: Block) -> str:
    """Breadcrumb line that makes a chunk self-contained when read on its own."""
    parts = [document.short_title]
    if block.chapter:
        parts.append(block.chapter)
    parts.append(block.section)
    return " / ".join(parts)


def tail_overlap(text: str, size: int) -> str:
    """Tail of roughly `size` characters, used as overlap for the next chunk.

    Whole sentences are preferred. When none fits — typical for "solid"
    paragraphs without sentence punctuation — we fall back to the trailing
    words. The result is hard-capped: an unbounded overlap leaks into the
    next chunk and makes chunk sizes snowball.
    """
    limit = int(size * 1.5)

    picked: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)):
        if total + len(sentence) > limit:
            break
        picked.insert(0, sentence)
        total += len(sentence) + 1
        if total >= size:
            break

    if picked:
        return " ".join(picked).strip()

    # No whole sentence fits — cut the tail on a word boundary instead.
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1:].strip() if space != -1 else ""


def render_header(base_header: str, extra_articles: list[str]) -> str:
    """Chunk header; lists every article when the chunk covers several short ones."""
    if not extra_articles:
        return base_header
    return f"{base_header} (а також статті {', '.join(extra_articles)})"


@dataclass
class Chunk:
    base_header: str
    body: str
    chapter: str | None
    section: str
    article_no: str | None
    sections: list[str]
    extra_articles: list[str]
    has_overlap: bool

    @property
    def header(self) -> str:
        return render_header(self.base_header, self.extra_articles)

    @property
    def text(self) -> str:
        return f"{self.header}\n\n{self.body}"


def chunk_document(document: Document) -> list[Chunk]:
    """Pack a document's blocks into chunks.

    Rules:
      * a chapter boundary always breaks the chunk;
      * an article boundary breaks it once the chunk has reached MIN_CHARS
        (adjacent short articles are merged into one meaningful chunk);
      * inside an article the break is size-driven, with OVERLAP_CHARS overlap;
      * a sentence is never split, except by the last-resort hard_split.
    """
    chunks: list[Chunk] = []
    if not document.blocks:
        return chunks

    current: Chunk | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.body.strip():
            chunks.append(current)
        current = None

    def start(block: Block, seed: str = "", overlapped: bool = False) -> None:
        nonlocal current
        current = Chunk(
            base_header=build_header(document, block),
            body=seed,
            chapter=block.chapter,
            section=block.section,
            article_no=block.article_no,
            sections=[block.section],
            extra_articles=[],
            has_overlap=overlapped,
        )

    for block in document.blocks:
        if current is None:
            start(block)
        elif block.chapter != current.chapter:
            flush()
            start(block)
        elif block.section != current.section and len(current.body) >= MIN_CHARS:
            flush()
            start(block)
        elif block.section != current.section:
            # A short article joins the previous chunk within the same chapter,
            # but only if the longer header still leaves room for the text.
            label = block.article_no or block.section
            extras = current.extra_articles + [label]
            if MAX_CHARS - len(render_header(current.base_header, extras)) - 2 < len(
                current.body
            ):
                flush()
                start(block)
            else:
                current.sections.append(block.section)
                current.extra_articles = extras

        assert current is not None
        # The header may grow if the chunk has to be restarted from this block,
        # so budget against the longer of the two possible headers.
        budget = MAX_CHARS - max(
            len(current.header), len(build_header(document, block))
        ) - 2

        for sentence in split_sentences(block.text):
            for piece in hard_split(sentence, budget):
                candidate = (
                    f"{current.body} {piece}".strip() if current.body else piece
                )
                if len(candidate) <= budget:
                    current.body = candidate
                    continue

                # chunk is full — emit it and open the next one
                previous_body = current.body
                previous_section = current.section
                flush()
                seed = ""
                overlapped = False
                if block.section == previous_section:
                    seed = tail_overlap(previous_body, OVERLAP_CHARS)
                    overlapped = bool(seed)
                start(block, seed=seed, overlapped=overlapped)
                assert current is not None
                budget = MAX_CHARS - len(current.header) - 2
                if current.body and len(current.body) + 1 + len(piece) > budget:
                    # overlap does not fit alongside the piece — drop the overlap
                    current.body = ""
                    current.has_overlap = False
                current.body = (
                    f"{current.body} {piece}".strip() if current.body else piece
                )

    flush()

    # Append a very short trailing chunk to the previous one when there is room.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.body) < 200
            and merged[-1].chapter == chunk.chapter
            and len(merged[-1].text) + len(chunk.body) + 1 <= MAX_CHARS
        ):
            merged[-1].body = f"{merged[-1].body} {chunk.body}".strip()
            for section in chunk.sections:
                if section not in merged[-1].sections:
                    merged[-1].sections.append(section)
            continue
        merged.append(chunk)

    return merged


# --------------------------------------------------------------------------
# 4. Assembling the knowledge base
# --------------------------------------------------------------------------


def drop_title_echo(blocks: list[Block], title: str) -> list[Block]:
    """Remove masthead paragraphs that merely repeat the act's title."""
    banned = {
        title.lower().strip("«»\" "),
        title.lower().replace("закон україни", "").strip("«»\" "),
        "закон україни",
        "кодекс законів про працю україни",
        "верховна рада україни",
    }
    return [
        block
        for block in blocks
        if block.text.lower().strip("«»\" .") not in banned
    ]


def load_documents() -> list[Document]:
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "data/raw/manifest.json not found — run "
            "python scripts/download_sources.py first"
        )

    documents: list[Document] = []
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        source_file = ROOT / entry["source_file"]
        raw_html = source_file.read_text(encoding="utf-8")
        paragraphs = html_to_paragraphs(extract_article_html(raw_html))

        documents.append(
            Document(
                document_id=entry["document_id"],
                title=entry["title"],
                short_title=entry["short_title"],
                document_type=entry["document_type"],
                source_file=entry["source_file"],
                source_url=entry["source_url"],
                retrieved_at=entry["retrieved_at"],
                blocks=drop_title_echo(parse_structure(paragraphs), entry["title"]),
            )
        )

    return documents


def to_record(document: Document, chunk: Chunk, index: int, total: int) -> dict:
    text = chunk.text
    return {
        "chunk_id": f"{document.document_id}_chunk_{index:04d}",
        "text": text,
        "metadata": {
            "document_id": document.document_id,
            "source_file": document.source_file,
            "source_url": document.source_url,
            "source_type": "html",
            "title": document.title,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "sections": chunk.sections,
            "article_no": chunk.article_no,
            "chunk_index": index,
            "chunks_in_document": total,
            "char_count": len(text),
            "has_overlap_with_previous": chunk.has_overlap,
            "language": LANGUAGE,
            "domain": DOMAIN,
            "document_type": document.document_type,
            "jurisdiction": JURISDICTION,
            "retrieved_at": document.retrieved_at,
            "processed_at": date.today().isoformat(),
        },
    }


def main() -> None:
    documents = load_documents()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "chunks.jsonl"

    records: list[dict] = []
    per_document: list[dict] = []

    for document in documents:
        normalized = write_normalized_markdown(document)
        chunks = chunk_document(document)
        total = len(chunks)
        document_records = [
            to_record(document, chunk, index, total)
            for index, chunk in enumerate(chunks)
        ]
        records.extend(document_records)

        sizes = [record["metadata"]["char_count"] for record in document_records]
        per_document.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_file": document.source_file,
                "source_url": document.source_url,
                "paragraphs": len(document.blocks),
                "chunks": total,
                "avg_chars": round(statistics.mean(sizes), 1) if sizes else 0,
                "min_chars": min(sizes, default=0),
                "max_chars": max(sizes, default=0),
                "normalized_file": str(normalized.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        print(
            f"[ok] {document.document_id:30s} "
            f"paragraphs={len(document.blocks):5d} "
            f"chunks={total:5d} "
            f"avg={per_document[-1]['avg_chars']:6.1f}"
        )

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_sizes = [record["metadata"]["char_count"] for record in records]
    stats = {
        "generated_at": date.today().isoformat(),
        "chunking": {
            "target_chars": TARGET_CHARS,
            "max_chars": MAX_CHARS,
            "min_chars": MIN_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "strategy": "structure-aware: chapter -> article -> sentence, "
                        "plus sentence-level overlap",
        },
        "totals": {
            "documents": len(documents),
            "chunks": len(records),
            "avg_chars": round(statistics.mean(all_sizes), 1),
            "median_chars": round(statistics.median(all_sizes), 1),
            "min_chars": min(all_sizes),
            "max_chars": max(all_sizes),
            "chunks_over_max": sum(1 for size in all_sizes if size > MAX_CHARS),
            "chunks_under_500": sum(1 for size in all_sizes if size < 500),
            "chunks_with_overlap": sum(
                1
                for record in records
                if record["metadata"]["has_overlap_with_previous"]
            ),
        },
        "documents": per_document,
    }
    (PROCESSED_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    totals = stats["totals"]
    print(
        f"\nWrote {len(records):,} chunks -> {output_path.relative_to(ROOT)}\n"
        f"Average size: {totals['avg_chars']} chars, "
        f"median: {totals['median_chars']}, "
        f"over {MAX_CHARS}: {totals['chunks_over_max']}, "
        f"under 500: {totals['chunks_under_500']}"
    )


if __name__ == "__main__":
    main()
