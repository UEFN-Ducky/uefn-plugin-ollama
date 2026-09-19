"""Real ollama.com library search (same filters as the public search page)."""

from __future__ import annotations

import json
import os
import re
import time
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

_SEARCH = "https://ollama.com/search"
_LIBRARY = "https://ollama.com/library"
_CAPS = ("vision", "tools", "thinking", "embedding", "cloud", "audio")
_CACHE_TTL_S = 30 * 60.0
_TAG_TTL_S = 6 * 3600.0


def _cache_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TMP") or ".") / "UEFN-Ducky"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ollama_library_cache.json"


def _cache_get(key: str, ttl: float) -> Any | None:
    try:
        raw = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    row = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(row, dict):
        return None
    if time.time() - float(row.get("t") or 0) > ttl:
        return None
    return row.get("v")


def _cache_put(key: str, value: Any) -> None:
    path = _cache_path()
    try:
        bag = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(bag, dict):
            bag = {}
    except (OSError, json.JSONDecodeError, TypeError):
        bag = {}
    bag[key] = {"t": time.time(), "v": value}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bag, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _strip_tags(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _slug_from_href(href: str) -> str:
    path = (href or "").split("?", 1)[0].strip()
    if path.startswith("/library/"):
        path = path[len("/library/") :]
    return path.strip("/").strip()


def _caps_from_text(text: str) -> list[str]:
    low = f" {text.lower()} "
    return [c for c in _CAPS if f" {c} " in low]


def _sizes_from_text(text: str) -> list[str]:
    found = re.findall(r"\b(\d+(?:\.\d+)?[bB]|e2b|e4b)\b", text, flags=re.I)
    out: list[str] = []
    seen: set[str] = set()
    for raw in found:
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _pulls_from_text(text: str) -> str:
    m = re.search(r"([\d.]+[KMB]?)\s*Pulls?", text, flags=re.I)
    return (m.group(1) if m else "").strip()


def _tag_count_from_text(text: str) -> int:
    m = re.search(r"(\d+)\s*Tags?", text, flags=re.I)
    return int(m.group(1)) if m else 0


def _updated_from_text(text: str) -> str:
    m = re.search(r"Updated\s+(.+?)(?:\s{2,}|$)", text, flags=re.I)
    return (m.group(1) if m else "").strip()


def parse_search_html(html: str) -> list[dict[str, Any]]:
    """Parse ollama.com/search cards. Slug + description + capability chips."""
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="(/library/[^"]+)"[^>]*>(.*?)</a>', html or "", flags=re.I | re.S):
        slug = _slug_from_href(m.group(1))
        if not slug or slug in seen or slug.startswith("?"):
            continue
        inner = _strip_tags(m.group(2))
        if not inner:
            continue
        seen.add(slug)
        title = slug.split("/")[-1]
        hm = re.search(rf"\b{re.escape(title)}\b", inner, flags=re.I)
        desc = inner
        if hm:
            desc = inner[hm.end() :].strip()
        desc = re.split(r"\b(?:vision|tools|thinking|embedding|cloud|audio)\b", desc, maxsplit=1)[0].strip()
        desc = re.sub(r"\s+", " ", desc)
        cards.append(
            {
                "slug": slug,
                "name": title,
                "description": desc[:280],
                "capabilities": _caps_from_text(inner),
                "sizes": _sizes_from_text(inner),
                "pulls": _pulls_from_text(inner),
                "tag_count": _tag_count_from_text(inner),
                "updated": _updated_from_text(inner),
            }
        )
    return cards


def _rows_from_json(data: Any) -> list[dict[str, Any]] | None:
    if isinstance(data, dict):
        rows = data.get("models") or data.get("items") or data.get("results")
    elif isinstance(data, list):
        rows = data
    else:
        return None
    if not isinstance(rows, list) or not rows:
        return None
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or row.get("name") or row.get("model") or "").strip()
        if not slug:
            continue
        caps = row.get("capabilities") or row.get("categories") or []
        if not isinstance(caps, list):
            caps = []
        sizes = row.get("sizes") or row.get("parameter_sizes") or []
        if not isinstance(sizes, list):
            sizes = []
        out.append(
            {
                "slug": slug.lstrip("/"),
                "name": str(row.get("display_name") or slug.split("/")[-1]),
                "description": str(row.get("description") or "")[:280],
                "capabilities": [str(c).lower() for c in caps if str(c).strip()],
                "sizes": [str(s).lower() for s in sizes if str(s).strip()],
                "pulls": str(row.get("pulls") or row.get("pull_count") or ""),
                "tag_count": int(row.get("tag_count") or row.get("tags") or 0)
                if str(row.get("tag_count") or row.get("tags") or "").isdigit()
                else 0,
                "updated": str(row.get("updated") or row.get("updated_at") or ""),
            }
        )
    return out or None


def search_library(
    query: str = "",
    capabilities: list[str] | None = None,
    order: str = "popular",
) -> dict[str, Any]:
    """List real models from ollama.com/search. JSON Accept first, HTML fallback."""
    caps = [str(c).strip().lower() for c in (capabilities or []) if str(c).strip()]
    order_s = "newest" if str(order or "").strip().lower() == "newest" else "popular"
    q = str(query or "").strip()
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    for c in caps:
        if c in _CAPS:
            params.append(("c", c))
    if order_s != "popular":
        params.append(("o", order_s))
    cache_key = f"search:{urlencode(params)}"
    hit = _cache_get(cache_key, _CACHE_TTL_S)
    if isinstance(hit, list):
        return {"ok": True, "models": hit, "cached": True}

    import httpx

    url = f"{_SEARCH}?{urlencode(params)}" if params else _SEARCH
    headers = {"User-Agent": "UEFN-Ducky Ollama library", "Accept": "application/json, text/html"}
    r = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    models: list[dict[str, Any]] | None = None
    if "json" in ctype:
        try:
            models = _rows_from_json(r.json())
        except Exception:
            models = None
    if models is None:
        models = parse_search_html(r.text)
    _cache_put(cache_key, models)
    return {"ok": True, "models": models, "cached": False}


def parse_tags_html(html: str, slug: str) -> list[dict[str, Any]]:
    """Variant rows from /library/{slug}/tags (name, size, context)."""
    text = _strip_tags(html or "")
    rows: list[dict[str, Any]] = []
    base = (slug or "").strip().split("/")[-1]
    for m in re.finditer(
        rf"\b({re.escape(base)}:[A-Za-z0-9._-]+|{re.escape(base)})\b.*?(\d+(?:\.\d+)?\s*[KMG]B)?.*?(\d+(?:\.\d+)?[kKmM])?",
        text,
    ):
        name = m.group(1)
        rows.append(
            {
                "name": name,
                "size": (m.group(2) or "").strip(),
                "context": (m.group(3) or "").strip(),
            }
        )
    # Dedup keeping first.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def list_library_tags(slug: str) -> dict[str, Any]:
    name = (slug or "").strip().strip("/")
    if not name:
        return {"ok": False, "error": "Model slug required", "tags": []}
    cache_key = f"tags:{name}"
    hit = _cache_get(cache_key, _TAG_TTL_S)
    if isinstance(hit, list):
        return {"ok": True, "slug": name, "tags": hit, "cached": True}

    import httpx

    url = f"{_LIBRARY}/{name}/tags"
    r = httpx.get(
        url,
        headers={"User-Agent": "UEFN-Ducky Ollama library", "Accept": "text/html"},
        timeout=20.0,
        follow_redirects=True,
    )
    r.raise_for_status()
    tags = parse_tags_html(r.text, name)
    _cache_put(cache_key, tags)
    return {"ok": True, "slug": name, "tags": tags, "cached": False}
