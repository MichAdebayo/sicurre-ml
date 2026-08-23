"""Generate or verify the reviewed OpenAPI contract from the FastAPI app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving.app import app  # noqa: E402

DEFAULT_OUTPUT = Path("docs/api/openapi.yaml")


def render_openapi() -> str:
    """Return a stable YAML representation of the runtime OpenAPI schema."""

    return yaml.safe_dump(
        app.openapi(),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the checked-in contract differs from generated output.",
    )
    args = parser.parse_args()

    generated = render_openapi()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != generated:
            print(
                f"OpenAPI contract drift detected in {args.output}. "
                "Run `make openapi` and review the diff."
            )
            return 1
        print(f"OpenAPI contract is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(f"Generated OpenAPI contract: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
