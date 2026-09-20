"""Strip the host MCP catalog before a local model evaluates it.

The app still sends the full tool index (~28k tokens) for cloud gateways.
Ollama prompt-evals that dump for half an hour on 8GB. Cut it here.
"""

from __future__ import annotations

import re

_SLIM_HEADINGS = (
    (
        "## Tool index",
        "## Tool index (local — catalog omitted)\n"
        "Hundreds of MCP tools exist; names are not listed here.\n"
        "1. `ducky_find_tools(query)` — search by intent\n"
        "2. `ducky_get_tools(name=…)` or `pattern=…` — fetch schema\n"
        "3. `ducky_call_tool(name, arguments)` — run it\n"
        "Floor tools are already in tools[].\n",
    ),
    (
        "## MCP server instructions",
        "## MCP server instructions\n"
        "Floor tools are in tools[]. Other tools: "
        "`ducky_find_tools` → `ducky_get_tools` → `ducky_call_tool`.\n",
    ),
    (
        "## Enabled Store desktop plugins",
        "## Enabled Store desktop plugins\n"
        "READY tools work now. Use `ducky_find_tools` for schemas.\n",
    ),
    (
        "## Available skill packs",
        "## Available skill packs (lazy-loaded)\n"
        'Call skill_read_subskill("<pack_id>", "core") when needed.\n',
    ),
)


def slim_local_system(text: str) -> str:
    """Drop host catalog dumps. Headings stay, bodies go."""
    if not text:
        return text
    if "## Tool index" not in text and "## MCP server instructions" not in text:
        return text
    parts = re.split(r"(?=^## )", text, flags=re.M)
    out: list[str] = []
    for part in parts:
        head = part.split("\n", 1)[0]
        replaced = False
        for prefix, stub in _SLIM_HEADINGS:
            if head.startswith(prefix):
                out.append(stub if stub.endswith("\n") else stub + "\n")
                replaced = True
                break
        if not replaced:
            out.append(part)
    return "".join(out)
