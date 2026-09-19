"""ollama.com/search HTML parse — real catalog cards."""

from __future__ import annotations

try:
    from .library import parse_search_html, parse_tags_html
except ImportError:
    from library import parse_search_html, parse_tags_html

_HTML = """
<ul role="list">
  <li>
    <a href="/library/qwen3.6">
      <h2>qwen3.6</h2>
      <p>Qwen3.6 delivers substantial upgrades in agentic coding.</p>
      vision tools thinking 27b 35b 6.7M Pulls 35 Tags Updated 2 weeks ago
    </a>
  </li>
  <li>
    <a href="/library/muse-glimmer">
      <h2>muse-glimmer</h2>
      <p>Meta's latest open model built for always-on local agents.</p>
      vision tools thinking 30b 218.1K Pulls 15 Tags Updated 3 weeks ago
    </a>
  </li>
</ul>
"""

_TAGS = """
<div>
  qwen3.6:latest 18 GB 256K
  qwen3.6:27b 16 GB 128K
</div>
"""


def test_parse_search_cards() -> None:
    rows = parse_search_html(_HTML)
    slugs = [r["slug"] for r in rows]
    assert slugs == ["qwen3.6", "muse-glimmer"]
    qwen = rows[0]
    assert "thinking" in qwen["capabilities"]
    assert "vision" in qwen["capabilities"]
    assert "tools" in qwen["capabilities"]
    assert "27b" in qwen["sizes"]
    assert qwen["pulls"] == "6.7M"
    assert qwen["tag_count"] == 35
    assert "agentic coding" in qwen["description"]


def test_parse_tags_keeps_variants() -> None:
    tags = parse_tags_html(_TAGS, "qwen3.6")
    names = [t["name"] for t in tags]
    assert "qwen3.6:latest" in names
    assert "qwen3.6:27b" in names


if __name__ == "__main__":
    test_parse_search_cards()
    test_parse_tags_keeps_variants()
    print("ok")
