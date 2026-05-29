"""Sicurre ML — Inference API.

Start locally:
    uvicorn src.serving.app:app --reload --port 8000

Endpoints:
    POST /v1/classify          → run the full pipeline
    GET  /v1/health            → liveness probe
    GET  /v1/ready             → readiness probe (checks ONNX model is loaded)
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status

load_dotenv()
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.inference.pipeline import ClassificationResult, run_pipeline  # noqa: E402

app = FastAPI(
    title="Sicurre Inference API",
    description="Multi-stage phishing detection — rules + blocklist + ONNX + LLM",
    version="0.1.0",
)

_bearer = HTTPBearer(auto_error=True)


# ---------------------------------------------------------------------------
# Auth — simple shared bearer token
# ---------------------------------------------------------------------------

def _verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    expected = os.environ.get("INFERENCE_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INFERENCE_API_KEY not configured on server",
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Message to classify")
    use_virustotal: bool = Field(False, description="Enable VirusTotal enrichment (adds latency)")
    use_llm: bool = Field(True, description="Enable LLM stage")


class ClassifyResponse(BaseModel):
    verdict: str
    is_phishing: bool
    composite_score: float
    stage_scores: dict[str, float]
    stage_labels: dict[str, str]
    explanation: str
    llm_provider: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/ready", tags=["ops"])
def ready() -> dict[str, Any]:
    """Checks that the ONNX model is loaded and the session is ready."""
    try:
        from src.inference.onnx_classifier import _load_session_and_tokenizer
        _load_session_and_tokenizer()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {exc}",
        )


@app.post(
    "/v1/classify",
    response_model=ClassifyResponse,
    tags=["inference"],
    dependencies=[Depends(_verify_token)],
)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Run the full phishing detection pipeline on the provided text."""
    from src.inference.onnx_classifier import _load_session_and_tokenizer

    try:
        _load_session_and_tokenizer()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {exc}",
        )

    result: ClassificationResult = run_pipeline(
        text=request.text,
        use_virustotal=request.use_virustotal,
        use_llm=request.use_llm,
    )
    return ClassifyResponse(
        verdict=result.verdict,
        is_phishing=result.is_phishing,
        composite_score=result.composite_score,
        stage_scores=result.stage_scores,
        stage_labels=result.stage_labels,
        explanation=result.explanation,
        llm_provider=result.llm_provider,
    )
