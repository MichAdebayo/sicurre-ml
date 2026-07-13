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
from hmac import compare_digest
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, Security, status
from fastapi.responses import PlainTextResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode

load_dotenv()
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.inference.onnx_classifier import get_model_version  # noqa: E402
from src.inference.pipeline import ClassificationResult, run_pipeline  # noqa: E402
from src.serving.identity import deployment_manifest, response_identity_headers  # noqa: E402
from src.serving.rate_limit import service_rate_limiter  # noqa: E402
from src.serving.telemetry import (  # noqa: E402
    emit_classify_request_log,
    emit_operational_log,
    observe_classify_request,
    runtime_telemetry,
)

_production = os.getenv("DEPLOYMENT_ENV", "development").lower() == "production"

app = FastAPI(
    title="Sicurre Inference API",
    description="Multi-stage phishing detection — rules + blocklist + ONNX + LLM",
    version="0.1.0",
    docs_url=None if _production else "/docs",
    redoc_url=None if _production else "/redoc",
    openapi_url=None if _production else "/openapi.json",
)


def _configure_tracing() -> None:
    ratio = min(max(float(os.getenv("OTEL_TRACE_SAMPLE_RATIO", "0.05")), 0.0), 1.0)
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "sicurre-ml-inference",
                "deployment.environment.name": os.getenv("DEPLOYMENT_ENV", "production"),
            }
        ),
        sampler=ParentBased(TraceIdRatioBased(ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy:4317"),
                insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
            )
        )
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=r"(^|/)(health|ready|metrics|docs|redoc|openapi\.json)$",
    )


_configure_tracing()

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
    if not compare_digest(credentials.credentials, expected):
        runtime_telemetry.observe_auth_failure()
        emit_operational_log("authentication_failure", category="invalid_bearer", status_code=401)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def _enforce_rate_limit() -> None:
    allowed, retry_after = service_rate_limiter.check()
    if not allowed:
        runtime_telemetry.observe_rate_limit()
        emit_operational_log("rate_limit_rejected", category="service_limit", status_code=429)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    subject: str = Field(default="", max_length=300, description="Email subject")
    sender: str = Field(default="", max_length=320, description="Email sender address")
    text: str = Field(..., min_length=1, max_length=4096, description="Email message body")
    use_virustotal: bool = Field(False, description="Enable VirusTotal enrichment (adds latency)")
    use_llm: bool = Field(True, description="Enable LLM stage")


class ClassifyResponse(BaseModel):
    verdict: str
    label_verdict: str
    is_phishing: bool
    composite_score: float
    stage_scores: dict[str, float]
    stage_labels: dict[str, str]
    label_distribution: dict[str, float]
    stage_breakdown: dict[str, dict[str, Any]]
    explanation: str
    llm_provider: str


def _public_stage_breakdown(
    stage_breakdown: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    hidden_fields = {"applied_weight", "contribution"}
    return {
        stage_name: {
            key: value
            for key, value in details.items()
            if key not in hidden_fields
        }
        for stage_name, details in stage_breakdown.items()
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
@app.get("/v1/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        runtime_telemetry.to_prometheus(model_version=get_model_version()),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/v1/ready", tags=["ops"])
def ready() -> dict[str, Any]:
    """Checks that the ONNX model is loaded and the session is ready."""
    try:
        from src.inference.onnx_classifier import _load_session_and_tokenizer
        _load_session_and_tokenizer()
        runtime_telemetry.set_model_ready(True)
        return {"status": "ready"}
    except Exception:
        runtime_telemetry.set_model_ready(False)
        emit_operational_log("model_readiness", category="model_not_ready", status_code=503)
        with trace.get_tracer(__name__).start_as_current_span(
            "model.readiness.failure"
        ) as span:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error.type", "model_not_ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
            headers={"Retry-After": "5"},
        ) from None


@app.post(
    "/v1/classify",
    response_model=ClassifyResponse,
    tags=["inference"],
    dependencies=[Depends(_verify_token)],
)
def classify(
    request: ClassifyRequest,
    http_response: Response,
    _: None = Depends(_enforce_rate_limit),
) -> ClassifyResponse:
    """Run the full phishing detection pipeline on the provided text."""
    import time

    from src.inference.onnx_classifier import _load_session_and_tokenizer

    started_at = time.perf_counter()
    try:
        _load_session_and_tokenizer()
    except Exception:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        observe_classify_request(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            latency_ms=latency_ms,
            model_version=get_model_version(),
            error_type="model_not_ready",
            mode="llm" if request.use_llm else "local",
        )
        emit_classify_request_log(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            latency_ms=latency_ms,
            model_version=get_model_version(),
            error_type="model_not_ready",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready",
            headers={"Retry-After": "5"},
        ) from None

    try:
        result: ClassificationResult = run_pipeline(
            text=request.text,
            subject=request.subject,
            sender=request.sender,
            use_virustotal=request.use_virustotal,
            use_llm=request.use_llm,
        )
    except Exception:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        observe_classify_request(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            latency_ms=latency_ms,
            model_version=get_model_version(),
            error_type="pipeline_unexpected",
            mode="llm" if request.use_llm else "local",
        )
        emit_operational_log(
            "classify_failure",
            category="pipeline_unexpected",
            status_code=500,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference pipeline failed",
        ) from None
    response = ClassifyResponse(
        verdict=result.verdict,
        label_verdict=result.label_verdict,
        is_phishing=result.is_phishing,
        composite_score=result.composite_score,
        stage_scores=result.stage_scores,
        stage_labels=result.stage_labels,
        label_distribution=result.label_distribution,
        stage_breakdown=_public_stage_breakdown(result.stage_breakdown),
        explanation=result.explanation,
        llm_provider=result.llm_provider,
    )

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    observe_classify_request(
        status_code=status.HTTP_200_OK,
        latency_ms=latency_ms,
        verdict=response.verdict,
        label_verdict=response.label_verdict,
        label_distribution=response.label_distribution,
        stage_latencies_ms=result.stage_latencies_ms,
        llm_provider=response.llm_provider,
        model_version=get_model_version(),
        mode="llm" if request.use_llm else "local",
    )
    emit_classify_request_log(
        status_code=status.HTTP_200_OK,
        latency_ms=latency_ms,
        verdict=response.verdict,
        label_verdict=response.label_verdict,
        label_distribution=response.label_distribution,
        stage_latencies_ms=result.stage_latencies_ms,
        llm_provider=response.llm_provider,
        model_version=get_model_version(),
    )
    for header, value in response_identity_headers().items():
        http_response.headers[header] = value
    return response


@app.get(
    "/v1/manifest",
    tags=["ops"],
    dependencies=[Depends(_verify_token)],
)
def manifest() -> dict[str, Any]:
    return deployment_manifest()
