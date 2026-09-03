"""Bind dashboard semantics to real probe and alert configuration."""

import json
from pathlib import Path

import yaml


def _panels() -> dict[str, dict]:
    dashboard = json.loads(Path("deploy/grafana/dashboards/sicurre-ml-runtime.json").read_text())
    panels = [panel for panel in dashboard["panels"] if panel["type"] != "row"]
    for row in (panel for panel in dashboard["panels"] if panel["type"] == "row"):
        panels.extend(row["panels"])
    return {panel["title"]: panel for panel in panels}


def test_public_health_is_an_http_probe_not_a_json_metrics_scrape() -> None:
    config = Path("deploy/alloy/config.alloy").read_text()
    module = config.split('prometheus.exporter.blackbox "sicurre_ml_health"')[1]
    definition = yaml.safe_load(module.split("`", 2)[1])["modules"]["health_json"]
    assert definition["prober"] == "http"
    assert definition["timeout"] == "5s"
    assert definition["http"]["valid_status_codes"] == [200]
    assert definition["http"]["follow_redirects"] is False
    assert definition["http"]["fail_if_body_not_matches_regexp"]
    assert 'address = sys.env("ML_HEALTH_PROBE_URL")' in module
    assert 'targets         = prometheus.exporter.blackbox.sicurre_ml_health.targets' in config
    assert 'metrics_path    = "/v1/health"' not in config
    assert 'metrics_path    = "/v1/metrics"' in config
    compose = yaml.safe_load(Path("docker-compose.prod.yml").read_text())
    assert compose["services"]["alloy"]["environment"]["ML_HEALTH_PROBE_URL"].endswith(
        "https://api.sicurre.com/v1/health}"
    )


def test_probe_cardinality_is_bounded_and_ci_never_probes_production() -> None:
    config = Path("deploy/alloy/config.alloy").read_text()
    relabel = config.split('prometheus.relabel "sicurre_ml_health"')[1].split("\n}", 1)[0]
    assert 'action = "keep"' in relabel
    names = relabel.split('regex = "')[1].split('"', 1)[0].split("|")
    assert len(names) == 5
    assert "probe_success" in names
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "ML_HEALTH_PROBE_URL=http://127.0.0.1:9/v1/health" in workflow


def test_health_panels_use_fresh_observations_and_distinct_unknown_states() -> None:
    panels = _panels()
    for title in ("Public API", "ML scrape", "Model", "Alloy"):
        panel = panels[title]
        expr = panel["targets"][0]["expr"]
        assert "timestamp(" in expr and "time() - 180" in expr
        assert "vector(3)" in expr
        mappings = panel["fieldConfig"]["defaults"]["mappings"][0]["options"]
        assert mappings["2"]["text"] == "Stale"
        assert mappings["3"]["text"] == "Unknown"
        assert mappings["3"]["color"] == "text"
    assert "probe_success" in panels["Public API"]["targets"][0]["expr"]
    assert 'service_name="sicurre-ml-health"' in panels["Public API"]["targets"][0]["expr"]


def test_percentiles_summarize_period_histograms_without_filling_empty_samples() -> None:
    panels = _panels()
    for title, quantile in (("P50", "0.5"), ("P95", "0.95")):
        panel = panels[title]
        expr = panel["targets"][0]["expr"]
        assert f"histogram_quantile({quantile}," in expr
        assert "sum by (le) (increase(" in expr
        assert "[$__range]" in expr
        assert "_count" in expr and " > 0" in expr
        assert "avg(" not in expr and "vector(0)" not in expr
        assert panel["options"]["colorMode"] == "none"
        assert panel["fieldConfig"]["defaults"]["unit"] == "ms"


def test_budget_colors_match_strict_greater_than_alert_boundaries() -> None:
    panels = _panels()
    series = panels["Active series / 3,000"]
    thresholds = series["fieldConfig"]["defaults"]["thresholds"]["steps"]
    alerts = json.loads(Path("deploy/grafana/alerts/sicurre-ml-alerts.json").read_text())
    by_uid = {alert["uid"]: alert for alert in alerts}
    assert series["options"]["colorMode"] == "value"
    assert thresholds[1]["value"] == by_uid["sicurre-ml-series-70"]["threshold"] + 1
    assert thresholds[2]["value"] == by_uid["sicurre-ml-series-85"]["threshold"] + 1
    assert thresholds[1]["color"] == "dark-orange" and thresholds[2]["color"] == "red"
    probe = by_uid["sicurre-ml-http-health"]
    assert "probe_success" in probe["expr"] and "timestamp(" in probe["expr"]
    assert probe["no_data"] == "Alerting" and probe["for"] == "2m"


def test_first_view_uses_roomy_cards_and_separate_diagnostic_sections() -> None:
    dashboard = json.loads(Path("deploy/grafana/dashboards/sicurre-ml-runtime.json").read_text())
    visible = [p for p in dashboard["panels"] if p["type"] != "row"]
    assert len({p["id"] for p in visible}) == len(visible)
    cards = [p for p in visible if p["type"] == "stat"]
    assert len(cards) == 4
    assert all(p["gridPos"]["w"] >= 6 and p["gridPos"]["h"] >= 4 for p in cards)
    rows = {p["title"]: p for p in dashboard["panels"] if p["type"] == "row"}
    assert rows["Service health"]["collapsed"] and rows["Resources"]["collapsed"]
    for panel in visible:
        if panel["type"] == "bargauge":
            assert panel["gridPos"]["h"] >= 4, "All three categories must remain visible"
            assert panel["options"]["valueMode"] == "text", "Values must contrast in both themes"
        if panel["type"] == "stat" and panel["options"]["colorMode"] == "value":
            assert panel["options"]["text"]["valueSize"] >= 24
    for index, panel in enumerate(visible):
        a = panel["gridPos"]
        for other in visible[index + 1:]:
            b = other["gridPos"]
            overlap = (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                       and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])
            assert not overlap, (panel["title"], other["title"])
