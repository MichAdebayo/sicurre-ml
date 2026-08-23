from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_openapi_generator_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "openapi.yaml"
    command = [
        sys.executable,
        ".github/scripts/generate_openapi.py",
        "--output",
        str(output),
    ]

    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)

    assert output.read_text(encoding="utf-8") == first
    subprocess.run([*command, "--check"], check=True)


def test_coverage_policy_enforces_full_and_core_thresholds(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "files": {
                    "src/config/settings.py": {
                        "summary": {"num_statements": 100, "covered_lines": 95}
                    },
                    "src/inference/pipeline.py": {
                        "summary": {"num_statements": 100, "covered_lines": 65}
                    },
                }
            }
        )
    )
    command = [
        sys.executable,
        ".github/scripts/check_coverage.py",
        "--report",
        str(report),
        "--full-min",
        "80",
        "--core-min",
        "90",
    ]

    assert subprocess.run(command, check=False).returncode == 0
    failing_command = command.copy()
    failing_command[failing_command.index("--full-min") + 1] = "81"
    assert subprocess.run(failing_command, check=False).returncode == 1
