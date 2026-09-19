"""Pull progress parsers + cancel marks the job done."""

from __future__ import annotations

try:
    from .progress import parse_pull_ndjson_line, parse_pull_tail
    from .pull import pull_cancel
except ImportError:
    from progress import parse_pull_ndjson_line, parse_pull_tail
    from pull import pull_cancel


def test_ndjson_percent() -> None:
    row = parse_pull_ndjson_line('{"status":"pulling layer","completed":50,"total":100}')
    assert row["percent"] == 50.0
    assert row["status"] == "pulling layer"
    done = parse_pull_ndjson_line('{"status":"success"}')
    assert done["done"] is True
    assert done["percent"] == 100.0
    err = parse_pull_ndjson_line('{"error":"no space"}')
    assert err["error"] == "no space"
    assert err["done"] is True


def test_tail_percent_and_success() -> None:
    mid = parse_pull_tail("pulling manifest\npulling sha256:abc 42%")
    assert mid["percent"] == 42.0
    done = parse_pull_tail("pulling sha256:abc 99%\nsuccess")
    assert done["done"] is True
    assert done["percent"] == 100.0


def test_cancel_missing_job() -> None:
    out = pull_cancel("missing")
    assert out["ok"] is True
    assert out["cancelled"] is True


if __name__ == "__main__":
    test_ndjson_percent()
    test_tail_percent_and_success()
    test_cancel_missing_job()
    print("ok")
