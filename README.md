<div align="center">

# Sicurre&nbsp;ML

**Three-class French email classifier — phishing · spam · legitimate**

Training pipeline, evaluation gates, and the ONNX inference service behind
[Sicurre](https://github.com/MichAdebayo/sicurre).

[![CI](https://img.shields.io/github/actions/workflow/status/MichAdebayo/sicurre-ml/ci.yml?branch=main&label=CI&logo=github)](https://github.com/MichAdebayo/sicurre-ml/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/MichAdebayo/sicurre-ml/cd.yml?branch=main&label=CD&logo=github)](https://github.com/MichAdebayo/sicurre-ml/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Base model](https://img.shields.io/badge/base-CamemBERTaV2-FCC624?logo=huggingface&logoColor=black)](docs/adr/0001-camembertv2-as-base-model.md)
[![Runtime](https://img.shields.io/badge/runtime-ONNX-005CED?logo=onnx&logoColor=white)](src/inference/onnx_classifier.py)
[![Tracking](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](docs/adr/0004-mlflow-for-tracking.md)
[![Weighted F1](https://img.shields.io/badge/weighted%20F1-0.8515-success)](docs/architecture/service-levels.md#the-incumbent)
[![Promotion](https://img.shields.io/badge/promotion-manual%20approval-important)](docs/model/promotion-policy.md)

</div>

---

## What this repo is

`sicurre-ml` owns **everything about the model** and nothing about the
application. It consumes frozen dataset exports produced by the companion
`sicurre` repo, trains a classifier, gates it against an immutable golden set,
and serves the approved artifact over an authenticated HTTP API.

| This repo owns | This repo does **not** own |
|----------------|----------------------------|
| Training configuration and metrics | Email ingestion and normalization |
| Evaluation and promotion gates | Dataset creation and lineage |
| MLflow tracking, HF publication | Application runtime and user data |
| The ONNX inference service | Remediation and notifications |

R2 is the dataset boundary between the two repos. Kaggle is the packaged
execution environment for training runs.

## System architecture

```mermaid
flowchart TB
    subgraph SIC["sicurre (companion repo)"]
        ING["Ingestion + normalization"] --> FROZEN["Frozen dataset export"]
    end

    FROZEN -->|"R2 — dataset boundary"| SYNC["Dataset sync"]

    subgraph ML["sicurre-ml"]
        SYNC --> KAGGLE["Kaggle training run<br/>CamemBERTaV2 · 4 epochs"]
        KAGGLE --> MLF["MLflow<br/>registry + metrics"]
        MLF --> CAND["Candidate<br/>1.0.N"]
        CAND --> EVAL{"Golden-set gate<br/>wF1 · recall · FP"}
        EVAL -->|fails| REJ["Rejected"]
        EVAL -->|passes| APPR["Human approval<br/>protected environment"]
        APPR --> HF["Hugging Face<br/>production tag"]
    end

    HF --> SERVE

    subgraph RUN["Inference service (Hetzner CX33)"]
        SERVE["ONNX Runtime session"] --> PIPE
        PIPE["4-stage pipeline"]
    end

    APP["sicurre app"] -->|"POST /v1/classify"| PIPE
    PIPE -->|"verdict + score"| APP
```

### The four-stage inference pipeline

A request is scored by four independent stages whose outputs are combined into
one weighted composite score.

```mermaid
flowchart LR
    IN["Email text"] --> N["Normalize"]
    N --> R["rules<br/><i>sub-ms</i>"]
    N --> B["blocklist<br/><i>sub-ms</i>"]
    N --> O["onnx<br/><i>&lt;500 ms</i>"]
    N -.->|"thread pool"| L["llm<br/><i>Groq → Cerebras</i>"]
    R --> C["Composite score"]
    B --> C
    O --> C
    L -.->|"may be absent"| C
    C --> V["verdict + degraded_reasons"]
```

The LLM stage is dispatched to a thread pool and joined before scoring, so it
gates total latency. **When the whole provider chain fails, the pipeline records
`llm_unavailable` and still returns a verdict from the remaining stages.** That
graceful degradation is why an LLM outage is a quality event, not an
availability incident — see [Service levels](docs/architecture/service-levels.md).

## Service levels

Full specification: **[docs/architecture/service-levels.md](docs/architecture/service-levels.md)**

| Plane | Objective | Target |
|-------|-----------|--------|
| Serving | Readiness (`/v1/ready` 200) | 99.0% / 30 d — **SLA 98.5%** |
| Serving | Request success (non-5xx) | 99.0% / 30 d |
| Serving | ONNX stage p95 | < 500 ms |
| Serving | Full-path p95 | < 8 s (LLM) · < 1 s (degraded) |
| Serving | Degraded-mode rate | < 10% |
| Model | Weighted F1 | ≥ incumbent (**0.8515**) |
| Model | Phishing recall | ≥ incumbent (**0.8810**) — never regress |
| Model | Legitimate false positives | ≤ incumbent (**8** of 42) |
| Pipeline | Lineage completeness | 100% of promotions |
| Pipeline | Rollback time | < 15 min |

The deployment is a single node with no failover, and the targets are set to be
honest about that rather than aspirational.

## API surface

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health`, `/v1/health` | — | Liveness |
| `GET` | `/v1/ready` | — | 200 when model loaded, 503 while downloading |
| `GET` | `/v1/metrics` | — | Plain-text metrics |
| `POST` | `/v1/classify` | Bearer | Classify an email |
| `GET` | `/v1/manifest` | Bearer | Deployed model identity |

Rate limiting defaults to 1 rps sustained with a burst of 5
(`INFERENCE_RATE_LIMIT_RPS`, `INFERENCE_RATE_LIMIT_BURST`).

## Model and training

| Property | Value |
|----------|-------|
| Base model | `almanach/camembertav2-base` |
| Labels | `phishing=0` · `spam=1` · `legitimate=2` |
| Max sequence length | 256 tokens |
| Epochs | **4** — standardised; see below |
| Batch size · LR | 8 · 2e-5 |
| Class weighting | `inverse_freq` |
| Serving format | ONNX |

**On the epoch count.** Four is the standing choice, matching the budget the
incumbent was trained with so candidates are comparable rather than
systematically undertrained. Three was tried on the 1 September corpus and
measured *significantly worse*: weighted F1 0.7141 against 0.8515, with
phishing recall identical at 0.8810 and legitimate false positives rising from
8 to 20 (McNemar p = 0.0018). Detection was unchanged; discrimination
collapsed. Do not lower it without repeating that measurement.

## Promotion

Training and promotion are **separate operations**. A finished run publishes an
immutable candidate and never advances a production pointer on its own.

1. Register in MLflow, alias `candidate`
2. Export ONNX, upload to Hugging Face, resolve the immutable commit SHA
3. Evaluate candidate *and* incumbent on the **same** golden-set version
4. Write a machine-readable promotion manifest
5. Human approves the protected `production` environment — the single manual act
6. Move MLflow alias and HF tag, pin and restart the server, validate, callback

Any post-approval failure restores every preserved pointer and sends a
`rolled_back` callback. Full policy: [docs/model/promotion-policy.md](docs/model/promotion-policy.md).

## Repository layout

| Path | Responsibility |
|------|----------------|
| `src/config/` | Training configuration, label constants |
| `src/data/` | Dataset loading, schema validation, tokenization |
| `src/model/` | Model loading, class weighting, trainer, metrics |
| `src/training/` | Training entrypoints |
| `src/evaluation/` | Golden-set registry, reports, error analysis |
| `src/inference/` | Four-stage pipeline, ONNX session, LLM chain |
| `src/serving/` | FastAPI app, auth, rate limiting |
| `src/registry/` | MLflow logging, Hugging Face publication |
| `deploy/` | Compose files, nginx, Grafana Alloy |
| `docs/` | Architecture, ADRs, runbooks — [index](docs/README.md) |

## Quickstart

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run uvicorn src.serving.app:app --reload --port 8000
```

## Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR, push | Lint, type-check, tests, coverage gate |
| `cd.yml` | Push to `main` | Build image, deploy to Hetzner |
| `train.yml` | Dispatch, dataset callback | Sync dataset, run Kaggle training |
| `evaluate-model.yml` | After training | Golden-set gate, candidate vs incumbent |
| `promote-model.yml` | Approval | Move production pointers, validate, callback |
| `release.yml` | Tag | Service release |

## Documentation

| Area | Entry point |
|------|-------------|
| Docs index | [docs/README.md](docs/README.md) |
| **Service levels (SLI/SLO/SLA)** | [docs/architecture/service-levels.md](docs/architecture/service-levels.md) |
| Promotion policy | [docs/model/promotion-policy.md](docs/model/promotion-policy.md) |
| Monitoring design | [docs/architecture/monitoring-design.md](docs/architecture/monitoring-design.md) |
| Architecture decisions | [docs/adr/](docs/adr/) |
| Runbooks | [docs/ops/](docs/ops/) |
| Deployment | [deploy/README.md](deploy/README.md) |
