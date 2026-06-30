"""Smoke test for cc_compute.py.

Run as ``python data/cc_compute_test.py`` from the paper folder.

Verifies:
1. The Claude Code CLI is on PATH.
2. A single round-trip returns non-empty text.
3. The call was logged to llm_calls.jsonl with the expected schema.
4. No `anthropic` SDK was imported anywhere in the data/ tree.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

from cc_compute import cc_call, RunLogger  # noqa: E402


def _smoke_dir() -> pathlib.Path:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = HERE / f"results_smoke_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _check_no_anthropic_imports() -> None:
    """Grep guard — no script may import the anthropic SDK."""
    matches: list[str] = []
    for path in HERE.rglob("*.py"):
        if path.name == "cc_compute_test.py":
            continue
        text = path.read_text()
        for ln, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import anthropic", "from anthropic")):
                matches.append(f"{path.relative_to(HERE)}:{ln}: {stripped}")
    if matches:
        print("FAIL: anthropic SDK imports are forbidden. Use cc_compute.py.")
        for m in matches:
            print("  " + m)
        sys.exit(1)


def main() -> None:
    print("→ Checking Claude Code CLI is on PATH...")
    if shutil.which("claude") is None:
        print("FAIL: 'claude' CLI not on PATH. Install Claude Code first.")
        sys.exit(1)
    print("  OK.")

    print("→ Checking no anthropic imports in data/...")
    _check_no_anthropic_imports()
    print("  OK.")

    run_dir = _smoke_dir()
    print(f"→ Issuing one Haiku round-trip into {run_dir}...")
    logger = RunLogger(run_dir)
    res = cc_call(
        "Reply with exactly the word 'ok'.",
        model="haiku",
        logger=logger,
        run_id="smoke-001",
        timeout_s=60.0,
    )

    print(f"  Response: {res.text!r}")
    assert res.text and len(res.text) > 0, "empty response"

    log_path = run_dir / "llm_calls.jsonl"
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1, f"expected 1 log line, got {len(lines)}"
    entry = json.loads(lines[0])
    for key in ("ts", "run_id", "model", "prompt", "response", "latency_ms"):
        assert key in entry, f"missing key in log entry: {key}"
    assert entry["run_id"] == "smoke-001"
    assert entry["response"] == res.text
    print("  OK — one valid log entry.")

    summary = {
        "smoke_passed": True,
        "model": res.model,
        "latency_ms": res.latency_ms,
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"→ Summary written: {run_dir / 'summary.json'}")
    print("Smoke complete.")


if __name__ == "__main__":
    main()
