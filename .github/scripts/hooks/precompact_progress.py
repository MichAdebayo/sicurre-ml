from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SESSION_UPDATES = ROOT / "tasks" / "session-updates.md"


def _load_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip()


def main() -> int:
    payload = _load_payload()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    branch = _run_git("branch", "--show-current") or "unknown"
    status = _run_git("status", "--short") or "clean"
    prompt_excerpt = str(
        payload.get("prompt") or payload.get("userPrompt") or ""
    ).strip()
    if prompt_excerpt:
        prompt_excerpt = prompt_excerpt.replace("\n", " ")[:240]
    else:
        prompt_excerpt = "not available"

    entry = "\n".join(
        [
            "",
            f"## {timestamp} PreCompact",
            f"- Branch: {branch}",
            f"- Prompt: {prompt_excerpt}",
            "- Git status:",
            "```text",
            status,
            "```",
            (
                "- Resume by reviewing `tasks/todo.md`, `tasks/lessons.md`, "
                "and the latest docs before coding."
            ),
            "",
        ]
    )

    SESSION_UPDATES.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_UPDATES.open("a", encoding="utf-8") as handle:
        handle.write(entry)

    print(
        json.dumps(
            {
                "continue": True,
                "systemMessage": (
                    "PreCompact session snapshot written to " "tasks/session-updates.md"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
