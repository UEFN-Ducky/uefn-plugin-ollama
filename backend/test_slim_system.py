"""Host catalog dumps are stripped in the plugin — not in the app."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slim_system import slim_local_system

FAT = """You are the UEFN Ducky agent.

## MCP server instructions
""" + ("unreal__PlaceDevice " * 200) + """
## Tool index (lazy — schemas via ducky_get_tools)
### core
""" + ("- `illustrator_create_document` — make a doc\n" * 400) + """
## Enabled Store desktop plugins (live MCP status)
- **Adobe** (`adobe`, 90 tools: illustrator_create_document, …)
## Available skill packs (lazy-loaded)
- `uefn` [shipped] — UEFN
  - `core`
  - `epic_mcp`
## Rules
Do the work.
"""


def test_slim_drops_catalog_keeps_rules():
    out = slim_local_system(FAT)
    assert "illustrator_create_document" not in out
    assert "unreal__PlaceDevice" not in out
    assert "ducky_find_tools" in out
    assert "## Rules" in out
    assert "Do the work." in out
    assert len(out) < 2000
    assert len(out) < len(FAT) // 10


def test_slim_noop_without_catalog():
    text = "You are a ducky.\n## Rules\nBe brief.\n"
    assert slim_local_system(text) == text


if __name__ == "__main__":
    test_slim_drops_catalog_keeps_rules()
    test_slim_noop_without_catalog()
    print("ok")
