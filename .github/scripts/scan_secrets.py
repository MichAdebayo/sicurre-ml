"""Standalone secret literal scanner for CI pipelines.

Scans all Python source files in the project for hardcoded credential patterns.
Mirrors the patterns defined in .github/scripts/hooks/stop_zero_errors.py so
both the local session hook and the CI workflow enforce the same rules.

Exit codes:
    0 — no violations found
    1 — one or more violations found (CI step fails)

Usage:
    python .github/scripts/scan_secrets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Repo root: two levels up from .github/scripts/
ROOT = Path(__file__).resolve().parents[2]

CHECK_DIRS = ["src", ".github/scripts", "tests", "ml", "mlops"]

# Keep these patterns in sync with stop_zero_errors.py.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'"hf_[A-Za-z0-9]{8,}"'),           # HuggingFace tokens
    re.compile(r'"ghp_[A-Za-z0-9]{8,}"'),           # GitHub personal access tokens
    re.compile(r'"gho_[A-Za-z0-9]{8,}"'),           # GitHub OAuth tokens
    re.compile(r'"ghs_[A-Za-z0-9]{8,}"'),           # GitHub Actions tokens
    re.compile(r'"sk-[A-Za-z0-9]{20,}"'),           # OpenAI API keys
    re.compile(r'"gsk_[A-Za-z0-9]{20,}"'),          # Groq API keys
    re.compile(r'"csk_[A-Za-z0-9]{20,}"'),          # Cerebras API keys
    re.compile(r'"dapi[A-Za-z0-9]{10,}"'),          # Databricks personal tokens
    re.compile(                                      # Generic credential assignments
        r'(?i)\b(?:password|passwd|api_key|secret_key|token|credential)'
        r'\s*=\s*["\'][^"\']{8,}["\']'
    ),
]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for dir_name in CHECK_DIRS:
        dir_path = ROOT / dir_name
        if not dir_path.exists():
            continue
        files.extend(
            py_file
            for py_file in dir_path.rglob("*.py")
            if ".venv" not in py_file.parts
        )
    return sorted(files)


def scan() -> list[str]:
    violations: list[str] = []
    for py_file in _python_files():
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                violations.append(
                    f"{py_file.relative_to(ROOT)}:{line_no}: "
                    f"potential hardcoded secret (pattern: {pattern.pattern!r})"
                )
    return violations


def main() -> int:
    if violations := scan():
        print(f"Secret scan FAILED — {len(violations)} violation(s) found:\n")
        for v in violations:
            print(f"  {v}")
        print(
            "\nRemove hardcoded credentials and load them via load_secrets() "
            "or environment variables instead."
        )
        return 1
    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
