from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECK_DIRS = ["src", ".github/scripts", "tests", "ml", "mlops"]

# Patterns that indicate a hardcoded secret literal in source code.
# Each pattern matches a known token prefix or a variable name commonly used
# for credentials followed by a non-empty string literal.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'"hf_[A-Za-z0-9]{8,}"'),          # HuggingFace tokens
    re.compile(r'"ghp_[A-Za-z0-9]{8,}"'),          # GitHub personal access tokens
    re.compile(r'"gho_[A-Za-z0-9]{8,}"'),          # GitHub OAuth tokens
    re.compile(r'"ghs_[A-Za-z0-9]{8,}"'),          # GitHub Actions tokens
    re.compile(r'"sk-[A-Za-z0-9]{20,}"'),          # OpenAI API keys
    re.compile(r'"dapi[A-Za-z0-9]{10,}"'),         # Databricks personal tokens
    re.compile(                                     # Generic credential assignments
        r'(?i)\b(?:password|passwd|api_key|secret_key|token|credential)\s*=\s*["\'][^"\']{8,}["\']'
    ),
]


def _existing_dirs() -> list[Path]:
    return [ROOT / name for name in CHECK_DIRS if (ROOT / name).exists()]


def _has_python_files(directory: Path) -> bool:
    return any(path.suffix in {".py", ".pyi"} for path in directory.rglob("*.py*"))


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in _existing_dirs():
        files.extend(
            path for path in directory.rglob("*.py") if ".venv" not in path.parts
        )
    return sorted(set(files))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def _python_executable() -> str:
    uv_python = ROOT / ".venv" / "bin" / "python"
    if uv_python.exists():
        return str(uv_python)
    return sys.executable


def _scan_for_secret_literals() -> list[str]:
    """Return a list of violation messages for any hardcoded secret literals found."""
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
                    f"Potential hardcoded secret in {py_file.relative_to(ROOT)}:{line_no} "
                    f"— matched pattern: {pattern.pattern!r}"
                )
    return violations


def main() -> int:
    errors: list[str] = []
    check_summaries: list[str] = []

    secret_violations = _scan_for_secret_literals()
    if secret_violations:
        errors.extend(secret_violations)
    else:
        check_summaries.append("secret scan passed")

    for path in _python_files():
        result = _run([_python_executable(), "-m", "py_compile", str(path)])
        if result.returncode != 0:
            errors.append(
                f"Syntax/compile error in {path.relative_to(ROOT)}\n{result.stderr.strip()}"
            )

    directories = [
        str(path.relative_to(ROOT))
        for path in _existing_dirs()
        if _has_python_files(path)
    ]
    if directories:
        result = _run(["uv", "run", "ruff", "check", *directories])
        if result.returncode != 0:
            errors.append((result.stdout or result.stderr).strip())
        else:
            check_summaries.append("ruff passed")

        result = _run(["uv", "run", "mypy", *directories])
        if result.returncode != 0:
            errors.append((result.stdout or result.stderr).strip())
        else:
            check_summaries.append("mypy passed")

    if errors:
        message = (
            "Zero-error stop check failed. Review the diagnostics below and invoke the "
            "Zero Error Remediator subagent before ending the session.\n\n"
            + "\n\n".join(errors)
        )
        print(
            json.dumps(
                {
                    "continue": False,
                    "stopReason": "Type or syntax errors remain",
                    "systemMessage": message,
                }
            )
        )
        return 2

    print(
        json.dumps(
            {
                "continue": True,
                "systemMessage": (
                    "Zero-error stop check passed. "
                    + ", ".join(check_summaries or ["compile checks passed"])
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
