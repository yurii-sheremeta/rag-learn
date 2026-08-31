"""Pipeline step 1: download the raw sources into data/raw/.

The sources are official texts of Ukrainian legal acts published on the
"Законодавство України" portal (zakon.rada.gov.ua). We request the /print
version of each page because it carries the full text of the act in a single
HTML document, without lazily loaded blocks.

Usage:  python scripts/download_sources.py
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.parse
import urllib.request
import zlib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE = "https://zakon.rada.gov.ua/laws/show/{rada_id}/print"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# document_id -> source description
SOURCES: dict[str, dict[str, str]] = {
    "kzpp": {
        "rada_id": "322-08",
        "title": "Кодекс законів про працю України",
        "short_title": "КЗпП України",
        "document_type": "code",
    },
    "zakon_pro_vidpustky": {
        "rada_id": "504/96-вр",
        "title": "Закон України «Про відпустки»",
        "short_title": "Про відпустки",
        "document_type": "law",
    },
    "zakon_pro_oplatu_praci": {
        "rada_id": "108/95-вр",
        "title": "Закон України «Про оплату праці»",
        "short_title": "Про оплату праці",
        "document_type": "law",
    },
    "zakon_pro_ohoronu_praci": {
        "rada_id": "2694-12",
        "title": "Закон України «Про охорону праці»",
        "short_title": "Про охорону праці",
        "document_type": "law",
    },
    "zakon_pro_voiennyi_stan": {
        "rada_id": "2136-20",
        "title": (
            "Закон України «Про організацію трудових відносин "
            "в умовах воєнного стану»"
        ),
        "short_title": "Трудові відносини в умовах воєнного стану",
        "document_type": "law",
    },
    "zakon_pro_kolektyvni_dohovory": {
        "rada_id": "3356-12",
        "title": "Закон України «Про колективні договори і угоди»",
        "short_title": "Про колективні договори і угоди",
        "document_type": "law",
    },
}


def build_url(rada_id: str) -> str:
    return BASE.format(rada_id=urllib.parse.quote(rada_id, safe="/-"))


def fetch(url: str) -> str:
    """Download a page and decompress gzip/deflate responses correctly.

    zakon.rada.gov.ua returns a compressed body even without an explicit
    Accept-Encoding request header, so we decompress manually based on the
    Content-Encoding response header.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "uk,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        encoding = (response.headers.get("Content-Encoding") or "").lower()

    if encoding == "gzip":
        payload = gzip.decompress(payload)
    elif encoding == "deflate":
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            payload = zlib.decompress(payload, -zlib.MAX_WBITS)

    return payload.decode("utf-8", errors="replace")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for document_id, meta in SOURCES.items():
        url = build_url(meta["rada_id"])
        target = RAW_DIR / f"{document_id}.html"
        print(f"[..] {document_id}: {url}")
        html = fetch(url)
        target.write_text(html, encoding="utf-8")
        print(f"[ok] {document_id}: {len(html):,} chars -> {target.relative_to(ROOT)}")

        manifest.append(
            {
                "document_id": document_id,
                "title": meta["title"],
                "short_title": meta["short_title"],
                "document_type": meta["document_type"],
                "source_url": url,
                "source_file": str(target.relative_to(ROOT)).replace("\\", "/"),
                "retrieved_at": date.today().isoformat(),
            }
        )
        time.sleep(1.0)  # polite delay between requests

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest saved: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
