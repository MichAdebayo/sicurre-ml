# Model Promotion Policy

## Decision

Training and production promotion are separate operations. A completed training
run registers and publishes an immutable **candidate**. It never advances an
MLflow production alias or Hugging Face production tag automatically.

Sicurre owns creation, review, versioning, and R2 publication of evaluation
datasets. Sicurre-ML only reads an approved immutable golden-set version and
records its identifier and checksum in evaluation evidence.

## Candidate gate

The candidate and incumbent must be evaluated against the same immutable
golden set. The initial synthetic set is explicitly provisional. Its results
demonstrate workflow integrity and non-regression; they do not establish
real-world performance.

The provisional gate requires:

- identical label and response contracts;
- candidate phishing recall no lower than the incumbent;
- candidate legitimate false-positive rate no higher than the incumbent;
- candidate weighted F1 no lower than the incumbent;
- human approval after reviewing the evidence and dataset provenance.

Latency is recorded as diagnostic MLflow evidence but is not a blocking gate
for the provisional version-one set.

Threshold tolerances are explicit inputs to the decision and are recorded in
the promotion manifest. An absent, invalid, or unreviewed golden set produces
an inconclusive result and cannot change production.

## Artifact lifecycle

1. Register the trained model in MLflow and assign its version the `candidate`
   alias.
2. Export ONNX locally before publication.
3. Upload the complete candidate artifact set to Hugging Face.
4. Resolve and record the resulting immutable Hugging Face commit SHA.
5. Evaluate candidate and incumbent on the approved golden set.
6. Create a machine-readable promotion manifest.
7. After explicit approval, move both production pointers to the approved
   immutable versions and verify them.
8. Preserve the previous production revisions for deterministic rollback.

## Protected approval and automatic completion

The passing evaluation invokes the reusable model-promotion workflow with only
immutable identifiers. The workflow job is bound to the protected GitHub
`production` environment. A repository owner reviews the referenced MLflow
evaluation and approves that environment deployment. That single approval is
the final human action; every subsequent operation is automatic.

The protected job must:

1. re-read MLflow evidence and reject anything except the exact passing
   evaluation/candidate tuple supplied by the evaluation workflow;
2. verify that the candidate Hugging Face SHA exists and contains the required
   ONNX artifact;
3. preserve the incumbent MLflow alias, Hugging Face tag, and deployed server
   identity;
4. mark the approved candidate as production in MLflow using both the
   `production` registered-model alias and bounded `sicurre.*` model-version
   and run tags;
5. move and verify the Hugging Face `production` tag;
6. pin the inference server to the candidate's immutable SHA and semantic model
   version, restart it, and validate health, readiness, authenticated inference,
   response identity, and deployment manifest identity;
7. send Sicurre the active deployment callback only after validation passes.

If any post-approval operation fails, the workflow restores all preserved
production pointers, restarts and validates the incumbent, then sends a
`rolled_back` callback. A failed or inconclusive evaluation never enters the
protected environment and never invokes promotion.

MLflow is the governance authority, Hugging Face is the artifact-delivery
authority, and the running deployment manifest is the runtime authority. Their
production identities must agree.

## Model identity convention

Candidate state is not encoded as a date and is not inferred from a run name.
Every trained artifact has one immutable semantic model identifier; promotion
changes its stage, not its version. Until an explicit release-series bump is
approved, the identifier is derived deterministically from the MLflow registry
version as `1.0.<registry-version>`.

The corresponding MLflow training run is named
`model-<semantic-version>-candidate`. The registered model version carries:

- `sicurre.model.semantic_version`;
- `sicurre.model.stage` (`candidate`, `production`, `retired`, or `rejected`);
- `sicurre.model.hf_revision`;
- `sicurre.evaluation.run_id`;
- `sicurre.promotion.github_run_id`, approver, and timestamp when promoted.

MLflow run tags use those dotted names. Unity Catalog reserves `.` in
registered-model tag keys, so the same model-version tags are stored with
underscores (for example `sicurre_model_stage` and
`sicurre_model_semantic_version`).

The registered model's `production` alias is the primary MLflow UI selector.
Dates remain timestamps only and are not model versions.

Hugging Face `main` is an artifact publication branch, not a production
selector. Runtime inference resolves the configured `production` revision to
an immutable SHA and downloads that SHA.

## Required lineage

Promotion evidence ties together:

- service source revision and GitHub Actions run;
- frozen training dataset ID, version tag, and checksum;
- golden-set ID, version, checksum, schema, provenance, and review status;
- MLflow run ID and registered model version;
- independent semantic model version;
- immutable Hugging Face commit SHA;
- incumbent and candidate metrics;
- gate result, thresholds, approver, and timestamp.

No secret, raw email content, generated sample text, or user PII belongs in the
manifest or MLflow tags.
