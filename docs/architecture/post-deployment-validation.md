# Post-deployment validation

Nothing about a deployment is trusted on the strength of the image having
started. Every deploy and every promotion runs
`deploy/scripts/validate_deployment.py` against the running container, and the
calling workflow stops on the first failure.

This document exists because the check was previously invisible: it ran on every
deployment but appeared in no README or runbook, so a reader had no way to know
it existed, and a report could not cite it.

## What it checks, in order

| Step | What it establishes |
|------|---------------------|
| `GET /v1/health`, up to 60 attempts | The process is answering at all |
| `GET /v1/ready`, up to 180 attempts | The ONNX session finished loading. The longer budget is deliberate — the model may still be downloading, and a slow start is not a failure |
| `POST /v1/classify`, authenticated | A real classification succeeds with a bearer token, not merely that the route is mounted |
| Response contract fields | The response carries the fields the public contract promises |
| Verdict membership | The verdict is inside the published set, so an unknown label cannot reach a caller |
| Identity headers | The response identifies which model produced it |
| `GET /v1/manifest`, authenticated | `schema_version` is 1 and the reported model identity matches what was deployed |

The manifest check is the one that makes the others meaningful: it confirms the
container is serving the revision the workflow intended, rather than a stale
image that happens to be healthy.

## Where it runs

**`cd.yml`** runs it after deploying to the host. A failure fails the deploy.

**`promote-model.yml`** runs it after pinning a newly promoted model. A failure
there triggers the rollback path — MLflow alias and Hugging Face tag restored
from the preserved values, the incumbent restarted and revalidated, and a
`rolled_back` status reported to Sicurre. This is why a failed promotion leaves
production on the previous model rather than in a partial state.

## Running it by hand

Against a container started locally:

```bash
INFERENCE_BASE_URL=http://127.0.0.1:8000 \
INFERENCE_API_KEY="$INFERENCE_API_KEY" \
uv run python deploy/scripts/validate_deployment.py
```

It exits non-zero on the first failed check and names which one, so it works as
a smoke test after any local change to the serving path.

## The companion check

`deploy/scripts/validate_observability.py` verifies that a request produces a
privacy-safe trace and an authentication log — that telemetry is emitted, and
that it carries neither credentials nor message content. Observability that
leaks what it observes is worse than none, so the check asserts the absence as
well as the presence.

## What it does not establish

It exercises the inference service in isolation. It says nothing about email
delivery, about the Cloudflare Worker, or about classification accuracy — a
verdict inside the contract is a well-formed verdict, not a correct one.
