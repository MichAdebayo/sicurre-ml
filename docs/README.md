# sicurre-ml documentation

Documentation for the model layer of Sicurre: training, evaluation, promotion,
and the ONNX inference service. The companion `sicurre` repo documents
ingestion, the data platform, and the application runtime.

Start at the [root README](../README.md) for the system diagram and a bounded
summary of the available evidence.

## Read these first

| Document | Why it matters |
|----------|----------------|
| [Performance and quality](architecture/performance.md) | What the system actually measures, and what has to be true before a new model ships |
| [Promotion policy](model/promotion-policy.md) | How a candidate becomes production, and what makes a promotion refuseable |
| [Monitoring design](architecture/monitoring-design.md) | What is observed at runtime and why each signal was chosen |

## Architecture

| Document | Scope |
|----------|-------|
| [Performance and quality](architecture/performance.md) | Measured latency, method limitations, configured budgets, and model evaluation evidence |
| [Deployment identity](architecture/deployment-identity.md) | Readable model version to immutable HF SHA mapping, runtime snapshot, and missing lineage |
| [Component design](architecture/component-design.md) | Module boundaries and the split with the companion repo |
| [Non-functional requirements](architecture/non-functional-requirements.md) | Engineering requirements, distinct from verified results and contractual promises |
| [Monitoring design](architecture/monitoring-design.md) | Runtime signals, log schema, alerting targets |
| [Sync contracts](architecture/sync-contracts.md) | Dataset and callback contracts across the repo boundary |
| [Post-deployment validation](architecture/post-deployment-validation.md) | What every deploy and promotion checks before the change is accepted |
| [API contract and coverage policy](architecture/api-contract-and-coverage-policy.md) | Deterministic OpenAPI generation, coverage gates |

## Model

| Document | Scope |
|----------|-------|
| [Promotion policy](model/promotion-policy.md) | Candidate gate, artifact lifecycle, protected approval, lineage |
| [Training plan](model/training-plan.md) | Objective, training path, data assumptions |

## Decisions

Architecture decision records live in [adr/](adr/), one file per decision,
each stating context, decision, and consequences.

| ADR | Decision |
|-----|----------|
| [0001](adr/0001-camembertv2-as-base-model.md) | CamemBERTaV2 as the base model |
| [0002](adr/0002-weightedtrainer-focal-loss.md) | Weighted trainer and loss strategy |
| [0003](adr/0003-kaggle-as-training-runtime.md) | Kaggle as the training runtime |
| [0004](adr/0004-mlflow-for-tracking.md) | MLflow for experiment tracking |
| [0005](adr/0005-huggingface-as-model-store.md) | Hugging Face as the model store |
| [0006](adr/0006-deployment-target-selection.md) | Deployment target |

## Operations

Runbooks live in [ops/](ops/). Reach for these during an incident.

| Runbook | Use when |
|---------|----------|
| [HF promotion runbook](ops/hf-promotion-runbook.md) | Moving or verifying the production tag |
| [Kaggle runbook](ops/kaggle-runbook.md) | A training run needs inspection or restart |
| [Monitoring](ops/monitoring.md) | Wiring or reading the observability stack |
| [Inference reliability remediation](ops/inference-reliability-remediation.md) | The service is degraded or failing |
| [Production readiness hardening](ops/production-readiness-hardening.md) | Pre-release checks |

## Certification

[simplon/simplon-coverage-checklist.md](simplon/simplon-coverage-checklist.md)
maps deliverables to competency criteria. Per-competency notes are in
[report/](report/).

## Boundaries

- `sicurre` owns ingestion, normalization, frozen dataset export, lineage, and
  the app runtime.
- `sicurre-ml` consumes frozen exports and produces versioned model artifacts.
- **R2** is the canonical dataset boundary.
- **Kaggle** is the packaged execution environment for training runs.
- **MLflow** is the governance authority, **Hugging Face** the artifact-delivery
  authority, and the running deployment manifest the runtime authority. Their
  production identities must agree.

## Configuration

[`.env.example`](../.env.example) is the configuration reference. It lists every
environment variable the service reads, grouped and commented, and is kept in step
with the code rather than duplicated into prose that would drift. Values that
change behaviour rather than credentials — the LLM chain budget, the per-provider
timeout, the rate limit, the stage fusion weights — are documented there with
their defaults.

Production values are not always the code defaults. Where a difference matters it
is named in [performance and quality](architecture/performance.md); `LLM_PROVIDER_TIMEOUT_SECONDS` is 4.5 s in production against a 2.5 s default, for
instance.

## Conventions

- Operational runbooks and architecture contracts stay in the repo.
- Secrets, tokens, and provider credentials never enter version control.
- Dataset samples are private unless explicitly sanitized for documentation.
- Measured numbers belong in [performance](architecture/performance.md) and are
  referenced from elsewhere rather than independently maintained in several pages.
