from __future__ import annotations

from fastapi.testclient import TestClient

from src.inference.pipeline import ClassificationResult
from src.serving import app as serving_app


def test_health_alias(monkeypatch) -> None:
    client = TestClient(serving_app.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint(monkeypatch) -> None:
    client = TestClient(serving_app.app)

    response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert "sicurre_inference_requests_total" in response.text


def test_classify_auth_and_public_contract(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.inference.onnx_classifier._load_session_and_tokenizer",
        lambda: None,
    )
    monkeypatch.setattr(
        serving_app,
        "run_pipeline",
        lambda **kwargs: ClassificationResult(
            verdict="safe",
            label_verdict="legitimate",
            composite_score=0.1,
            is_phishing=False,
            stage_latencies_ms={"onnx": 1.0},
            stage_scores={"onnx": 0.1},
            stage_labels={"onnx": "legitimate"},
            label_distribution={"phishing": 0.1, "spam": 0.2, "legitimate": 0.7},
            stage_breakdown={
                "onnx": {
                    "active": True,
                    "configured_weight": 0.2,
                    "reason": "Base model output",
                    "predicted_label": "legitimate",
                    "confidence": 0.9,
                    "applied_weight": 1.0,
                    "contribution": 0.1,
                }
            },
            explanation="",
            llm_provider="",
        ),
    )

    client = TestClient(serving_app.app)
    response = client.post(
        "/v1/classify",
        headers={"Authorization": "Bearer test-key"},
        json={
            "subject": "Bonjour",
            "sender": "contact@example.com",
            "text": "hello",
            "use_virustotal": False,
            "use_llm": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "stage_weights_configured" not in body
    assert "stage_weights_applied" not in body
    assert "stage_contributions" not in body
    assert "applied_weight" not in body["stage_breakdown"]["onnx"]
    assert "contribution" not in body["stage_breakdown"]["onnx"]
    assert response.headers["X-Sicurre-Service-Version"] == "0.1.0"
    assert response.headers["X-Sicurre-Model-Version"]
    assert response.headers["X-Sicurre-Model-Revision"]
    assert response.headers["X-Sicurre-Deployment-Revision"]


def test_classify_accepts_worker_payload_boundaries(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.inference.onnx_classifier._load_session_and_tokenizer",
        lambda: None,
    )
    monkeypatch.setattr(
        serving_app,
        "run_pipeline",
        lambda **kwargs: ClassificationResult(
            verdict="safe",
            label_verdict="legitimate",
            composite_score=0.1,
            is_phishing=False,
        ),
    )

    response = TestClient(serving_app.app).post(
        "/v1/classify",
        headers={"Authorization": "Bearer test-key"},
        json={"subject": "S" * 500, "text": "T" * 5500, "use_llm": False},
    )

    assert response.status_code == 200


def test_rate_limit_returns_retry_after(monkeypatch) -> None:
    from src.serving.rate_limit import service_rate_limiter

    service_rate_limiter.reset()
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("INFERENCE_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("INFERENCE_RATE_LIMIT_BURST", "1")
    monkeypatch.setattr(
        "src.inference.onnx_classifier._load_session_and_tokenizer",
        lambda: None,
    )
    monkeypatch.setattr(
        serving_app,
        "run_pipeline",
        lambda **kwargs: ClassificationResult(
            verdict="safe",
            label_verdict="legitimate",
            composite_score=0.1,
            is_phishing=False,
        ),
    )
    client = TestClient(serving_app.app)
    payload = {"text": "hello", "use_llm": False}

    assert client.post(
        "/v1/classify", headers={"Authorization": "Bearer test-key"}, json=payload
    ).status_code == 200
    limited = client.post(
        "/v1/classify", headers={"Authorization": "Bearer test-key"}, json=payload
    )

    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_unexpected_pipeline_error_is_sanitized(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.inference.onnx_classifier._load_session_and_tokenizer",
        lambda: None,
    )

    def fail_pipeline(**kwargs: object) -> ClassificationResult:
        raise RuntimeError("raw message content must not escape")

    monkeypatch.setattr(serving_app, "run_pipeline", fail_pipeline)
    client = TestClient(serving_app.app)
    response = client.post(
        "/v1/classify",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "private email body", "use_llm": False},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Inference pipeline failed"}
    assert "private email body" not in response.text


def test_manifest_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    client = TestClient(serving_app.app)

    assert client.get("/v1/manifest").status_code == 401
    response = client.get(
        "/v1/manifest", headers={"Authorization": "Bearer test-key"}
    )

    assert response.status_code == 200
    assert response.json()["service"]["api_contract"] == "v1"


def test_untrusted_host_is_rejected() -> None:
    client = TestClient(serving_app.app)

    response = client.get("/v1/health", headers={"Host": "attacker.invalid"})

    assert response.status_code == 400
