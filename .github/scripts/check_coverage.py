"""Enforce honest statement coverage thresholds from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CORE_PREFIXES = (
    "src/config/",
    "src/data/",
    "src/model/",
    "src/training/",
)


def _statement_totals(
    files: dict[str, dict[str, Any]],
    prefixes: tuple[str, ...] | None = None,
) -> tuple[int, int]:
    statements = 0
    covered = 0
    for raw_path, payload in files.items():
        path = raw_path.replace("\\", "/")
        if prefixes is not None and not path.startswith(prefixes):
            continue
        summary = payload["summary"]
        statements += int(summary["num_statements"])
        covered += int(summary["covered_lines"])
    return covered, statements


def _percentage(covered: int, statements: int) -> float:
    return 100.0 if statements == 0 else covered * 100.0 / statements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("coverage.json"))
    parser.add_argument("--full-min", type=float, default=80.0)
    parser.add_argument("--core-min", type=float, default=90.0)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    files: dict[str, dict[str, Any]] = payload["files"]
    full_covered, full_statements = _statement_totals(files)
    core_covered, core_statements = _statement_totals(files, CORE_PREFIXES)
    full_percent = _percentage(full_covered, full_statements)
    core_percent = _percentage(core_covered, core_statements)

    print(
        f"Full source statement coverage: {full_percent:.2f}% "
        f"({full_covered}/{full_statements})"
    )
    print(
        f"Training core statement coverage: {core_percent:.2f}% "
        f"({core_covered}/{core_statements})"
    )

    failures: list[str] = []
    if full_percent < args.full_min:
        failures.append(f"full source {full_percent:.2f}% < {args.full_min:.2f}%")
    if core_percent < args.core_min:
        failures.append(f"training core {core_percent:.2f}% < {args.core_min:.2f}%")
    if failures:
        print("Coverage gate failed: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
