# Deployment identity

## Model version maps to an immutable revision

Yes: the deployment manifest pairs a readable model version with its immutable
Hugging Face commit SHA. It should retain both, not replace the version with
the SHA. The authenticated `GET /v1/manifest` endpoint currently exposes them as
`model.version` and `model.revision`.

Observed production mapping on 2 September 2026:

```text
phishing-detector-1.0.0 -> 86e90dc5ff205a9711afe5de696f3d81ab8432be
```

This is an observation of the running service, not a new version assignment.
The version label is supplied through `MODEL_VERSION`; it can be a historical
or default value. It must be checked against registry and promotion evidence
before claiming that it uniquely identifies a release. The promotion policy
derives labels such as `1.0.15` from registry version 15, but that is not the
label observed on this running deployment. Do not silently relabel the snapshot
to make it match the policy; reconcile registry and deployment evidence first.

## Field meanings

The schema is [deployment-manifest.schema.json](../../deploy/deployment-manifest.schema.json).
Schema version 1 and existing field names are preserved.

| Field | Identity |
|-------|----------|
| `service.version` | Service/code release, independent of the trained model |
| `service.api_contract` | API contract version, currently `v1` |
| `model.version` | Human-readable model version label |
| `model.requested_revision` | Configured HF reference; may be a mutable tag locally, but should be pinned to the SHA for deployment |
| `model.revision` | Resolved HF commit SHA reported by the loaded model runtime |
| `dataset.version` | Frozen training dataset version, not the evaluation-set version or model SHA |
| `deployment.revision` | Deployment/build revision, normally the Git commit used as the image tag |
| `deployment.container_image_digest` | Immutable container image digest, distinct from the Git and HF commits |

The full SHA is needed for artifact traceability. A dashboard can show a
shortened value such as `86e90dc5...8432be`, while preserving the full revision
in the manifest and a detail view. Keep the model version tag visible too.

The identity headers mirror the manifest:

| Header | Manifest field |
|--------|----------------|
| `X-Sicurre-Service-Version` | `service.version` |
| `X-Sicurre-Model-Version` | `model.version` |
| `X-Sicurre-Model-Revision` | `model.revision` |
| `X-Sicurre-Deployment-Revision` | `deployment.revision` |

## Verified runtime snapshot and gaps

The [2 September observation file](evidence/inference-smoke-2026-09-02.json)
retains the local and production manifests together with the measured request
timings. It is dated evidence, not a production configuration file or a pointer
that should be updated automatically.

Production reported service version `0.2.0`, a pinned model requested revision,
and an immutable container digest. Local development reported service version
`0.1.0`, requested model reference `production`, deployment revision `latest`,
and no container digest. Both resolved to the same full model SHA.

Both reported `dataset.version="unknown"`. This is a provenance gap, not a
dataset version of zero. Resolving it requires the actual frozen training
dataset reference from training/registry evidence and consistent deployment
configuration. Substituting a model SHA, a guessed version, or the golden-set
version would create false lineage.

The API snapshot alone is not independent verification of the registry, model
files, or image bytes. Reading it does not repair missing lineage. No production
environment values, pointers, model versions, or API fields were changed in
this documentation pass.

## Optional display names

A friendly model name could be added later as presentation metadata assigned
once per immutable model SHA. It must not become a replacement for the model
version or SHA, and must remain stable when the same artifact is redeployed.

This is different from the mutable MLflow `production` alias, which identifies
the currently promoted registry entry. No friendly-name assignment mechanism
or new alias field is implemented by this documentation update.
