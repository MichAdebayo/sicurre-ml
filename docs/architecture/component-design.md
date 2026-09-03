# Component Design

## Repository role

`sicurre-ml` owns the model layer for Sicurre end to end:

- loading frozen training datasets
- tokenization and dataset preparation
- model training and class weighting
- evaluation against immutable golden sets, and promotion gating
- experiment tracking and model publication
- the ONNX inference service that serves the promoted artifact

## Module map

| Area | Responsibility |
|------|----------------|
| `src/config/` | Training configuration, label constants, run settings |
| `src/data/` | Dataset loading, schema validation, tokenization helpers |
| `src/model/` | Model loading, class weighting, trainer logic, metrics |
| `src/training/` | End-to-end training entrypoints and argument construction |
| `src/evaluation/` | Golden-set registry, evaluation reports, error analysis |
| `src/inference/` | Four-stage classification pipeline, ONNX session, LLM chain |
| `src/serving/` | FastAPI application, authentication, rate limiting |
| `src/registry/` | MLflow logging and Hugging Face publication |
| `scripts/` | Local operational helpers |

## Inference composition

`src/inference/pipeline.py` scores a request through four independent stages
and combines them into one weighted composite:

| Stage | Source | Character |
|-------|--------|-----------|
| `rules` | `rules.py` | Deterministic URL heuristics, main thread |
| `blocklist` | `blocklist.py`, `phishtank_loader.py` | Local PhishTank lookup, main thread. Optional VirusTotal enrichment (`use_virustotal`, default off) makes a live API call with a 10 s timeout |
| `onnx` | `onnx_classifier.py` | The trained model, the primary signal, dispatched to a thread pool |
| `llm` | `llm_classifier.py` | Mistral then Groq in a thread pool, with configurable network timeouts and an intended chain budget. This is not a strict wall-clock deadline. Cerebras is implemented but not in the chain |

`onnx` and `llm` are submitted to their executors first; `rules` and then
`blocklist` run sequentially on the main thread before the futures are joined.
Limited local and production smoke measurements are documented in
[performance](performance.md); they do not establish end-to-end delivery time
or sustained capacity.

Stage weights are environment-driven so they can be tuned without a redeploy.

The LLM stage is joined before the composite is formed, so it gates total
latency. When the whole provider chain fails the pipeline records
`llm_unavailable` in `degraded_reasons` and still returns a verdict from the
remaining stages. That degradation is deliberate and is tracked as its own
separate operational signal rather than automatically as an HTTP failure; see
[performance](performance.md).

## Branch discipline

| Branch | Focus |
|--------|-------|
| `mlops` | Orchestration, dataset sync, promotion automation, deployment |
| `main` | Released state; deployment follows it |

Work uses contextual prefixes such as `feat/`, `fix/`, `perf/`, `chore/`, or
`docs/` and follows the review chain `mlops` -> `develop` -> `main`.

## Boundary with sicurre

The companion `sicurre` repo is the system of record for operational data and
dataset creation. This repo consumes frozen exports only and must not
re-implement the data platform. R2 is the boundary; Kaggle is the packaged
execution environment.
