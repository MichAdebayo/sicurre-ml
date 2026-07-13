from __future__ import annotations

from fastapi.testclient import TestClient

from src.inference.pipeline import ClassificationResult
from src.serving import app as serving_app


def test_classify_response_hides_internal_weight_fields(monkeypatch) -> None:
    def fake_ready() -> None:
        return None

    captured: dict[str, str] = {}

    def fake_run_pipeline(
        *,
        text: str,
        subject: str | None,
        sender: str | None,
        use_virustotal: bool,
        use_llm: bool,
    ) -> ClassificationResult:
        captured["subject"] = subject or ""
        captured["sender"] = sender or ""
        captured["text"] = text
        return ClassificationResult(
            verdict="safe",
            label_verdict="legitimate",
            composite_score=0.12,
            is_phishing=False,
            stage_scores={"onnx": 0.12},
            stage_labels={"onnx": "legitimate"},
            label_distribution={"phishing": 0.12, "spam": 0.18, "legitimate": 0.7},
            stage_weights_configured={"onnx": 0.2, "llm": 0.45},
            stage_weights_applied={"onnx": 1.0},
            stage_contributions={"onnx": 0.12},
            stage_breakdown={
                "onnx": {
                    "active": True,
                    "configured_weight": 0.2,
                    "reason": "Base model output",
                    "predicted_label": "legitimate",
                    "confidence": 0.7,
                    "applied_weight": 1.0,
                    "contribution": 0.12,
                }
            },
            explanation="",
            llm_provider="",
        )

    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr("src.inference.onnx_classifier._load_session_and_tokenizer", fake_ready)
    monkeypatch.setattr(serving_app, "run_pipeline", fake_run_pipeline)

    client = TestClient(serving_app.app)
    response = client.post(
        "/v1/classify",
        headers={"Authorization": "Bearer test-key"},
        json={
            "subject": "Action immediate requise",
            "sender": "support@paypa1-security.com",
            "text": "Suspicious message",
            "use_virustotal": False,
            "use_llm": False,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert captured == {
        "subject": "Action immediate requise",
        "sender": "support@paypa1-security.com",
        "text": "Suspicious message",
    }

    assert "stage_weights_configured" not in body
    assert "stage_weights_applied" not in body
    assert "stage_contributions" not in body
    assert "applied_weight" not in body["stage_breakdown"]["onnx"]
    assert "contribution" not in body["stage_breakdown"]["onnx"]