"""Alloy must keep its read positions across a deploy.

`loki.source.docker` records how far it has read into each container's log.
That state lives under Alloy's storage path, and the storage path lived inside
the container's writable layer — which `docker compose up --force-recreate`
destroys on every deploy.

So each deploy left Alloy with no memory of what it had already shipped, and it
re-read the app's retained json-file history (5 files x 50 MB). Those entries
carry their original timestamps, so Loki rejected them as too old and
`loki_write_dropped_entries_total` climbed — 55 errors observed on 3 September.

The failure is invisible without this test: the pipeline is healthy, the logs
are shipped, and the only symptom is a counter and a gap where redelivered
entries were refused. Nothing about the config file looks wrong, because the
cause is a missing volume rather than a wrong setting.

Note this is not the write-ahead log. The WAL sits under the same storage path,
so it was destroyed on every restart too — which is why "truncate the WAL on
restart", the first-guess fix, would have changed nothing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path("docker-compose.prod.yml")


def _alloy() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["alloy"]


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_alloy_declares_an_explicit_storage_path() -> None:
    """Relying on the default puts state in the container's writable layer."""
    command = _alloy()["command"]
    assert "--storage.path=" in command, (
        "Alloy has no --storage.path, so its state defaults into the container "
        "filesystem and is destroyed by --force-recreate on every deploy"
    )


def test_the_storage_path_is_backed_by_a_named_volume() -> None:
    """A path is only persistent if something outlives the container at it."""
    alloy = _alloy()
    command = alloy["command"]
    storage = command.split("--storage.path=", 1)[1].split()[0].strip()

    mounts = [str(v) for v in alloy.get("volumes", [])]
    backing = [m for m in mounts if m.split(":")[1:2] == [storage]]

    assert backing, f"nothing is mounted at {storage}; state would not survive"

    source = backing[0].split(":")[0]
    assert not source.startswith((".", "/")), (
        f"{storage} is backed by bind mount {source!r}. A named volume is "
        f"required so the state is managed by Docker rather than depending on "
        f"a host path existing with the right ownership."
    )
    assert source in (_compose().get("volumes") or {}), (
        f"volume {source!r} is mounted but never declared"
    )


def test_the_state_mount_is_writable() -> None:
    """Alloy writes positions; a read-only mount would fail silently at runtime."""
    alloy = _alloy()
    storage = alloy["command"].split("--storage.path=", 1)[1].split()[0].strip()

    for mount in (str(v) for v in alloy.get("volumes", [])):
        parts = mount.split(":")
        if parts[1:2] == [storage]:
            assert "ro" not in parts[2:], f"{storage} is mounted read-only"


def test_the_app_still_bounds_its_log_retention() -> None:
    """The retained history is what gets re-shipped when positions are lost.

    Persistent positions are the fix, but an unbounded log would make any
    future position loss unbounded too. This keeps the blast radius finite.
    """
    app = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["app"]
    options = app["logging"]["options"]

    assert options.get("max-size"), "app logs must have a max-size"
    assert options.get("max-file"), "app logs must have a max-file"
