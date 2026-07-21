from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src.config.training_config import load_secrets
from src.evaluation.golden_set import (
    GoldenSetReference,
    evaluate_golden_set,
    load_approved_golden_set,
)
from src.evaluation.hub_onnx import HubTransformersPredictor
from src.evaluation.promotion import GoldenMetrics, decide_candidate_promotion
from src.evaluation.retrieval import download_r2_object
from src.registry.callbacks import post_provenance_callback

GOLDEN_VERSION = "golden-20260719-v1"
GOLDEN_SHA256 = "bc329213cacddab409a63deb9d663e593351b6e740a45cdada4c201e3beea346"
GOLDEN_KEY = "golden.jsonl"


def _required(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Required configuration is missing: {name}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mlflow-run-id", required=True)
    parser.add_argument("--candidate-mlflow-model-version", required=True)
    parser.add_argument("--candidate-hf-revision", required=True)
    parser.add_argument("--incumbent-hf-revision", required=True)
    parser.add_argument("--hf-repository", required=True)
    parser.add_argument("--semantic-version", required=True)
    parser.add_argument("--service-source-revision", required=True)
    parser.add_argument("--training-github-run-id", required=True)
    parser.add_argument("--training-dataset-version", required=True)
    return parser.parse_args()


def _gate_metrics(report: object) -> GoldenMetrics:
    return GoldenMetrics(
        weighted_f1=report.weighted_f1,  # type: ignore[attr-defined]
        phishing_recall=report.phishing_recall,  # type: ignore[attr-defined]
        legitimate_false_positive_rate=report.legitimate_false_positive_rate,  # type: ignore[attr-defined]
        p95_latency_ms=report.p95_latency_ms,  # type: ignore[attr-defined]
    )


def main() -> None:
    args = _parse_args()
    secrets = load_secrets("local")
    callback_base = _required(
        secrets["SICURRE_CALLBACK_BASE_URL"] or os.getenv("SICURRE_CALLBACK_BASE_URL"),
        "SICURRE_CALLBACK_BASE_URL",
    )
    callback_token = _required(
        secrets["SICURRE_INTERNAL_API_KEY"], "SICURRE_INTERNAL_API_KEY"
    )
    hf_token = secrets["HF_TOKEN"]
    from huggingface_hub import HfApi

    incumbent_revision = HfApi().model_info(
        repo_id=args.hf_repository,
        revision=args.incumbent_hf_revision,
        token=hf_token,
    ).sha
    if not incumbent_revision:
        raise RuntimeError("Incumbent revision did not resolve to an immutable SHA")

    with tempfile.TemporaryDirectory(prefix="sicurre-golden-") as temporary:
        root = Path(temporary)
        golden_path = download_r2_object(
            endpoint=_required(
                secrets["R2_EVALUATION_ENDPOINT"], "R2_EVALUATION_ENDPOINT"
            ),
            bucket=_required(
                secrets["R2_EVALUATION_BUCKET_NAME"],
                "R2_EVALUATION_BUCKET_NAME",
            ),
            object_key=GOLDEN_KEY,
            access_key_id=_required(
                secrets["R2_EVALUATION_ACCESS_KEY_ID"],
                "R2_EVALUATION_ACCESS_KEY_ID",
            ),
            secret_access_key=_required(
                secrets["R2_EVALUATION_SECRET_ACCESS_KEY"],
                "R2_EVALUATION_SECRET_ACCESS_KEY",
            ),
            destination=root / "golden.jsonl",
            expected_sha256=GOLDEN_SHA256,
        )
        samples = load_approved_golden_set(
            golden_path,
            GoldenSetReference(
                dataset_id="sicurre-golden",
                version=GOLDEN_VERSION,
                sha256=GOLDEN_SHA256,
                schema_version="1",
                provenance="synthetic_provisional",
                review_status="approved",
            ),
        )
        candidate = HubTransformersPredictor(
            repo_id=args.hf_repository,
            revision=args.candidate_hf_revision,
            token=hf_token,
        )
        candidate_report = evaluate_golden_set(samples, candidate.predict)
        print("Candidate golden-set inference complete.")
        del candidate
        gc.collect()
        incumbent = HubTransformersPredictor(
            repo_id=args.hf_repository,
            revision=incumbent_revision,
            token=hf_token,
        )
        incumbent_report = evaluate_golden_set(samples, incumbent.predict)
        print("Incumbent golden-set inference complete.")
        del incumbent
        gc.collect()
        decision = decide_candidate_promotion(
            _gate_metrics(candidate_report), _gate_metrics(incumbent_report)
        )

        import mlflow

        os.environ["DATABRICKS_HOST"] = _required(
            secrets["DATABRICKS_HOST"], "DATABRICKS_HOST"
        )
        os.environ["DATABRICKS_TOKEN"] = _required(
            secrets["DATABRICKS_TOKEN"], "DATABRICKS_TOKEN"
        )
        mlflow.set_tracking_uri("databricks")
        experiment = _required(
            secrets["MLFLOW_EXPERIMENT_NAME"], "MLFLOW_EXPERIMENT_NAME"
        )
        email = secrets["DATABRICKS_EMAIL"]
        evaluation_experiment = (
            f"/Users/{email}/{experiment}-golden-evaluation" if email else experiment
        )
        mlflow.set_experiment(evaluation_experiment)
        with mlflow.start_run(run_name=f"golden-{args.semantic_version}") as run:
            evaluation_run_id = run.info.run_id
            metrics = {
                "candidate_weighted_f1": candidate_report.weighted_f1,
                "production_weighted_f1": incumbent_report.weighted_f1,
                "candidate_phishing_recall": candidate_report.phishing_recall,
                "production_phishing_recall": incumbent_report.phishing_recall,
                "candidate_legitimate_false_positives": float(
                    candidate_report.legitimate_false_positive_count
                ),
                "production_legitimate_false_positives": float(
                    incumbent_report.legitimate_false_positive_count
                ),
                "candidate_p95_latency_ms": candidate_report.p95_latency_ms,
                "production_p95_latency_ms": incumbent_report.p95_latency_ms,
            }
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "sicurre.evaluation.outcome": decision.result,
                    "sicurre.golden_set.version": GOLDEN_VERSION,
                    "sicurre.golden_set.sha256": GOLDEN_SHA256,
                    "sicurre.candidate.run_id": args.candidate_mlflow_run_id,
                    "sicurre.candidate.hf_revision": args.candidate_hf_revision,
                    "sicurre.incumbent.hf_revision": incumbent_revision,
                }
            )
            evidence = {
                "candidate": candidate_report.to_dict(),
                "incumbent": incumbent_report.to_dict(),
                "decision": decision.to_dict(),
                "golden_set": {"version": GOLDEN_VERSION, "sha256": GOLDEN_SHA256},
            }
            evidence_path = root / "evaluation-evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            mlflow.log_artifact(str(evidence_path), artifact_path="evaluation")

        post_provenance_callback(
            base_url=callback_base,
            path="/internal/ml/candidates",
            bearer_token=callback_token,
            payload={
                "model_name": "sicurre-phishing-classifier",
                "semantic_version": args.semantic_version,
                "service_source_revision": args.service_source_revision,
                "mlflow_run_id": args.candidate_mlflow_run_id,
                "mlflow_model_version": args.candidate_mlflow_model_version,
                "huggingface_repository": args.hf_repository,
                "huggingface_revision": args.candidate_hf_revision,
                "training_github_run_id": args.training_github_run_id,
                "training_dataset_version_tag": args.training_dataset_version,
            },
        )
        evaluated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        outcome = {"pass": "passed", "fail": "failed"}.get(
            decision.result, "inconclusive"
        )
        post_provenance_callback(
            base_url=callback_base,
            path="/internal/ml/evaluations",
            bearer_token=callback_token,
            payload={
                "candidate_mlflow_run_id": args.candidate_mlflow_run_id,
                "incumbent_huggingface_revision": incumbent_revision,
                "evaluation_set_version_tag": GOLDEN_VERSION,
                "evaluation_set_checksum": GOLDEN_SHA256,
                "mlflow_evaluation_run_id": evaluation_run_id,
                "outcome": outcome,
                "metrics": {
                    key: int(value) if "false_positives" in key else value
                    for key, value in metrics.items()
                    if "latency" not in key
                },
                "evaluated_at": evaluated_at,
            },
        )
        print(f"Evaluation completed: {outcome}; MLflow run {evaluation_run_id}")


if __name__ == "__main__":
    main()
