"""Real Alloy/Prometheus probe regression; all traffic and storage stay local.

Run with ALLOY_BINARY and PROMETHEUS_BINARY pointing to the pinned test tools.
The normal CI Alloy job additionally validates the complete production graph.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(check: Callable[[], bool], timeout: int = 25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.25)
    raise AssertionError("Local monitoring condition did not become true")


def test_real_probe_rejects_bad_body_http_errors_and_recovers(tmp_path: Path) -> None:
    alloy = os.getenv("ALLOY_BINARY")
    prometheus = os.getenv("PROMETHEUS_BINARY")
    if not alloy or not prometheus:
        pytest.skip("Set ALLOY_BINARY and PROMETHEUS_BINARY for the isolated runtime test")

    @dataclass
    class Response:
        code: int = 200
        body: bytes = b'{"status":"ok"}'

    response = Response()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(response.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Location", "/v1/health")
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prom_port, alloy_port = _free_port(), _free_port()
    prom_url = f"http://127.0.0.1:{prom_port}"
    prom_config = tmp_path / "prometheus.yml"
    prom_config.write_text("global:\n  scrape_interval: 1s\nscrape_configs: []\n")
    source = Path("deploy/alloy/config.alloy").read_text()
    probe = source[source.index('prometheus.exporter.blackbox "sicurre_ml_health"'):]
    probe = probe[:probe.index('prometheus.remote_write "grafana_cloud"')]
    probe = probe.replace('scrape_interval = "60s"', 'scrape_interval = "10s"')
    config = tmp_path / "probe.alloy"
    config.write_text(probe + f'''
prometheus.remote_write "grafana_cloud" {{
  endpoint {{ url = "{prom_url}/api/v1/write" }}
  external_labels = {{ stack = "sicurre-ml" }}
}}
''')
    dashboard = json.loads(Path("deploy/grafana/dashboards/sicurre-ml-runtime.json").read_text())
    health = next(p for p in dashboard["panels"] if p["title"] == "Public API")
    expr = health["targets"][0]["expr"]

    def value(query: str, at: float | None = None) -> float | None:
        params = {"query": query}
        if at is not None:
            params["time"] = str(at)
        with urlopen(f"{prom_url}/api/v1/query?{urlencode(params)}", timeout=2) as result:
            rows = json.load(result)["data"]["result"]
        return float(rows[0]["value"][1]) if rows else None

    processes: list[subprocess.Popen[bytes]] = []
    try:
        with (tmp_path / "processes.log").open("w") as log:
            processes.append(subprocess.Popen([
                prometheus, f"--config.file={prom_config}",
                f"--storage.tsdb.path={tmp_path / 'tsdb'}",
                f"--web.listen-address=127.0.0.1:{prom_port}",
                "--web.enable-remote-write-receiver",
            ], stdout=log, stderr=log))
            _wait_for(lambda: value("vector(1)") == 1)
            assert value(expr) == 3, "Absent telemetry must not be healthy"
            processes.append(subprocess.Popen([
                alloy, "run", "--disable-reporting",
                f"--server.http.listen-addr=127.0.0.1:{alloy_port}",
                f"--storage.path={tmp_path / 'alloy'}", str(config),
            ], env={**os.environ, "ML_HEALTH_PROBE_URL":
                    f"http://127.0.0.1:{server.server_port}/v1/health"}, stdout=log, stderr=log))
            _wait_for(lambda: value(expr) == 1)
            response.body = b'{"status":"broken"}'
            _wait_for(lambda: value(expr) == 0)
            response.body = b'{"status":"ok"}'
            _wait_for(lambda: value(expr) == 1)
            for code in (503, 302):
                response.code = code
                _wait_for(lambda: value("probe_http_status_code") == code)
                assert value(expr) == 0
            response.code = 200
            _wait_for(lambda: value(expr) == 1)
            assert value('count({service_name="sicurre-ml-health"})') == 5
            server.shutdown()
            server.server_close()
            _wait_for(lambda: value(expr) == 0)
            processes[-1].terminate()
            processes[-1].wait(timeout=10)
            assert value(expr, time.time() + 190) == 2, "Old data must be stale"
            assert value(expr, time.time() + 400) == 3, "Expired data must be unknown"
    except AssertionError as error:
        logs = (tmp_path / "processes.log").read_text()[-5000:]
        raise AssertionError(f"{error}\n{logs}") from error
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
