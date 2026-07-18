import runpy
from pathlib import Path
from typing import Any, Callable, cast


def _decoder() -> Callable[..., dict[str, Any]]:
    namespace = runpy.run_path(str(Path("deploy/scripts/validate_deployment.py")))
    return cast(Callable[..., dict[str, Any]], namespace["_decode_json_body"])


def test_http_error_body_may_be_plain_text() -> None:
    decode = _decoder()

    assert decode(b"Invalid host header", status_code=400) == {
        "error": "non_json_http_error"
    }


def test_valid_json_object_is_preserved() -> None:
    decode = _decoder()

    assert decode(b'{"status":"ok"}', status_code=200) == {"status": "ok"}
