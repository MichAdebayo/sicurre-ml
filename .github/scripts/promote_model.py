#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.registry.callbacks import post_provenance_callback
from src.registry.tags import model_version_tag_key

MODEL_NAME = "main.sicurre.phishing-detector"
_SHA_RE = re.compile(r"^[a-f0-9]{40}$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True, slots=True)
class PromotionSnapshot:
    model_name: str
    candidate_model_version: str
    candidate_run_id: str
    candidate_hf_revision: str
    semantic_version: str
    previous_model_version: str | None
    previous_run_id: str | None
    previous_hf_revision: str | None
    previous_semantic_version: str | None
    previous_candidate_alias_version: str | None

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> PromotionSnapshot:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _required(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Required configuration is missing: {name}")
    return value


def _validated_sha(value: str, name: str) -> str:
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-character commit SHA")
    return value


def _validated_semver(value: str) -> str:
    if not _SEMVER_RE.fullmatch(value):
        raise ValueError("semantic_version must be valid SemVer")
    return value


def _write_actions_outputs(values: dict[str, str | None]) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value or ''}\n")


def _configure_mlflow() -> Any:
    _required(os.getenv("DATABRICKS_HOST"), "DATABRICKS_HOST")
    _required(os.getenv("DATABRICKS_TOKEN"), "DATABRICKS_TOKEN")
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.MlflowClient()


def _alias_version(client: Any, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception as exc:
        text = str(exc).lower()
        if "not found" in text or "resource_does_not_exist" in text:
            return None
        raise


def _set_run_tags(client: Any, run_id: str | None, tags: dict[str, str]) -> None:
    if not run_id:
        return
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def _set_version_tags(client: Any, version: str | None, tags: dict[str, str]) -> None:
    if not version:
        return
    for key, value in tags.items():
        client.set_model_version_tag(
            MODEL_NAME,
            version,
            model_version_tag_key(key),
            value,
        )


def _clear_candidate_promotion_tags(
    client: Any,
    *,
    run_id: str,
    model_version: str,
) -> None:
    for key in (
        "sicurre.promotion.github_run_id",
        "sicurre.promotion.approved_by",
        "sicurre.promotion.approved_at",
        "sicurre.promotion.completed_at",
    ):
        try:
            client.delete_tag(run_id, key)
        except Exception:
            pass
        try:
            client.delete_model_version_tag(
                MODEL_NAME,
                model_version,
                model_version_tag_key(key),
            )
        except Exception:
            pass


def _verify_evidence(
    client: Any,
    *,
    evaluation_run_id: str,
    candidate_run_id: str,
    candidate_model_version: str,
    candidate_hf_revision: str,
    hf_repository: str,
    semantic_version: str,
    incumbent_hf_revision: str,
) -> Any:
    evaluation = client.get_run(evaluation_run_id)
    tags = evaluation.data.tags
    expected = {
        "sicurre.evaluation.outcome": "pass",
        "sicurre.candidate.run_id": candidate_run_id,
        "sicurre.candidate.mlflow_model_version": candidate_model_version,
        "sicurre.candidate.hf_revision": candidate_hf_revision,
        "sicurre.candidate.hf_repository": hf_repository,
        "sicurre.model.semantic_version": semantic_version,
        "sicurre.incumbent.hf_revision": incumbent_hf_revision,
    }
    mismatches = [
        key for key, value in expected.items() if str(tags.get(key, "")) != value
    ]
    if mismatches:
        raise RuntimeError(
            "MLflow promotion evidence is missing or inconsistent: "
            + ", ".join(sorted(mismatches))
        )

    version = client.get_model_version(MODEL_NAME, candidate_model_version)
    if version.run_id != candidate_run_id:
        raise RuntimeError("MLflow candidate model version does not belong to the run")
    return version


def _verify_hf_candidate(repo_id: str, revision: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id=repo_id, revision=revision, token=token)
    if info.sha != revision:
        raise RuntimeError("Hugging Face candidate did not resolve to the approved SHA")
    files = set(api.list_repo_files(repo_id=repo_id, revision=revision, token=token))
    if "model.onnx" not in files:
        raise RuntimeError("Approved Hugging Face revision has no model.onnx artifact")


def _resolve_hf_production(repo_id: str, token: str | None) -> str | None:
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(
            repo_id=repo_id, revision="production", token=token
        ).sha
    except Exception as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            return None
        raise


def _move_hf_tag(
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.delete_tag(repo_id=repo_id, tag="production", token=token)
    except Exception as exc:
        if "404" not in str(exc) and "not found" not in str(exc).lower():
            raise
    if revision is None:
        return
    api.create_tag(
        repo_id=repo_id,
        tag="production",
        revision=revision,
        token=token,
    )
    resolved = api.model_info(
        repo_id=repo_id,
        revision="production",
        token=token,
    ).sha
    if resolved != revision:
        raise RuntimeError("Hugging Face production tag verification failed")


def _restore_registry(
    client: Any,
    snapshot: PromotionSnapshot,
    *,
    hf_repository: str,
    hf_token: str | None,
) -> None:
    _move_hf_tag(hf_repository, snapshot.previous_hf_revision, hf_token)
    if snapshot.previous_model_version:
        client.set_registered_model_alias(
            snapshot.model_name,
            "production",
            snapshot.previous_model_version,
        )
        _set_version_tags(
            client,
            snapshot.previous_model_version,
            {"sicurre.model.stage": "production"},
        )
        _set_run_tags(
            client,
            snapshot.previous_run_id,
            {"sicurre.model.stage": "production"},
        )
    else:
        try:
            client.delete_registered_model_alias(snapshot.model_name, "production")
        except Exception as exc:
            if "not found" not in str(exc).lower():
                raise
    if snapshot.previous_model_version != snapshot.candidate_model_version:
        _set_version_tags(
            client,
            snapshot.candidate_model_version,
            {"sicurre.model.stage": "candidate"},
        )
        _set_run_tags(
            client,
            snapshot.candidate_run_id,
            {"sicurre.model.stage": "candidate"},
        )
        _clear_candidate_promotion_tags(
            client,
            run_id=snapshot.candidate_run_id,
            model_version=snapshot.candidate_model_version,
        )
    if snapshot.previous_candidate_alias_version:
        client.set_registered_model_alias(
            snapshot.model_name,
            "candidate",
            snapshot.previous_candidate_alias_version,
        )


def promote(args: argparse.Namespace) -> None:
    semantic_version = _validated_semver(args.semantic_version)
    candidate_revision = _validated_sha(
        args.candidate_hf_revision, "candidate_hf_revision"
    )
    client = _configure_mlflow()
    hf_token = _required(os.getenv("HF_TOKEN"), "HF_TOKEN")
    previous = _alias_version(client, "production")
    previous_candidate = _alias_version(client, "candidate")
    previous_hf_revision = _resolve_hf_production(args.hf_repository, hf_token)
    if not previous or not previous_hf_revision:
        raise RuntimeError("Promotion requires a recorded production incumbent")
    _verify_evidence(
        client,
        evaluation_run_id=args.evaluation_run_id,
        candidate_run_id=args.candidate_mlflow_run_id,
        candidate_model_version=args.candidate_mlflow_model_version,
        candidate_hf_revision=candidate_revision,
        hf_repository=args.hf_repository,
        semantic_version=semantic_version,
        incumbent_hf_revision=previous_hf_revision,
    )
    _verify_hf_candidate(args.hf_repository, candidate_revision, hf_token)

    previous_tags = previous.tags or {}
    previous_semantic_version = previous_tags.get(
        model_version_tag_key("sicurre.model.semantic_version")
    ) or previous_tags.get(model_version_tag_key("sicurre.model.runtime_version"))
    snapshot = PromotionSnapshot(
        model_name=MODEL_NAME,
        candidate_model_version=args.candidate_mlflow_model_version,
        candidate_run_id=args.candidate_mlflow_run_id,
        candidate_hf_revision=candidate_revision,
        semantic_version=semantic_version,
        previous_model_version=str(previous.version) if previous else None,
        previous_run_id=str(previous.run_id) if previous else None,
        previous_hf_revision=previous_hf_revision,
        previous_semantic_version=previous_semantic_version,
        previous_candidate_alias_version=(
            str(previous_candidate.version) if previous_candidate else None
        ),
    )
    snapshot.write(Path(args.state_path))
    _write_actions_outputs(
        {
            "previous_hf_revision": snapshot.previous_hf_revision,
            "previous_semantic_version": snapshot.previous_semantic_version,
        }
    )

    promoted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    candidate_tags = {
        "sicurre.model.semantic_version": semantic_version,
        "sicurre.model.stage": "production",
        "sicurre.model.hf_revision": candidate_revision,
        "sicurre.evaluation.run_id": args.evaluation_run_id,
        "sicurre.promotion.github_run_id": args.github_run_id,
        "sicurre.promotion.approved_by": args.approved_by,
        "sicurre.promotion.approved_at": args.approved_at,
        "sicurre.promotion.completed_at": promoted_at,
    }
    previous_tags = {"sicurre.model.stage": "retired"}
    try:
        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            args.candidate_mlflow_model_version,
        )
        if (
            snapshot.previous_candidate_alias_version
            == args.candidate_mlflow_model_version
        ):
            client.delete_registered_model_alias(MODEL_NAME, "candidate")
        _set_version_tags(client, args.candidate_mlflow_model_version, candidate_tags)
        _set_run_tags(client, args.candidate_mlflow_run_id, candidate_tags)
        if snapshot.previous_model_version != args.candidate_mlflow_model_version:
            _set_version_tags(client, snapshot.previous_model_version, previous_tags)
            _set_run_tags(client, snapshot.previous_run_id, previous_tags)
        _move_hf_tag(args.hf_repository, candidate_revision, hf_token)
    except Exception:
        _restore_registry(
            client,
            snapshot,
            hf_repository=args.hf_repository,
            hf_token=hf_token,
        )
        raise
    print(
        f"Verified promotion pointers: {MODEL_NAME} v"
        f"{args.candidate_mlflow_model_version} / {args.hf_repository}@{candidate_revision}"
    )


def rollback(args: argparse.Namespace) -> None:
    snapshot = PromotionSnapshot.read(Path(args.state_path))
    client = _configure_mlflow()
    hf_token = _required(os.getenv("HF_TOKEN"), "HF_TOKEN")
    _restore_registry(
        client,
        snapshot,
        hf_repository=args.hf_repository,
        hf_token=hf_token,
    )
    print("MLflow and Hugging Face production pointers restored.")


def callback(args: argparse.Namespace) -> None:
    status = args.status
    deployed_revision = args.deployed_revision if status == "active" else None
    deployed_at = (
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if status == "active"
        else None
    )
    response = post_provenance_callback(
        base_url=_required(
            os.getenv("SICURRE_CALLBACK_BASE_URL"),
            "SICURRE_CALLBACK_BASE_URL",
        ),
        path="/internal/ml/deployments",
        bearer_token=_required(
            os.getenv("SICURRE_INTERNAL_API_KEY"), "SICURRE_INTERNAL_API_KEY"
        ),
        payload={
            "candidate_mlflow_run_id": args.candidate_mlflow_run_id,
            "mlflow_evaluation_run_id": args.evaluation_run_id,
            "github_run_id": args.github_run_id,
            "approved_by": args.approved_by,
            "approved_at": args.approved_at,
            "status": status,
            "deployed_revision": deployed_revision,
            "failure_reason": args.failure_reason or None,
            "deployed_at": deployed_at,
        },
    )
    print(
        f"Sicurre deployment callback: status={response.status}, "
        f"idempotent={str(response.idempotent).lower()}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("--evaluation-run-id", required=True)
    promote_parser.add_argument("--candidate-mlflow-run-id", required=True)
    promote_parser.add_argument("--candidate-mlflow-model-version", required=True)
    promote_parser.add_argument("--candidate-hf-revision", required=True)
    promote_parser.add_argument("--hf-repository", required=True)
    promote_parser.add_argument("--semantic-version", required=True)
    promote_parser.add_argument("--github-run-id", required=True)
    promote_parser.add_argument("--approved-by", required=True)
    promote_parser.add_argument("--approved-at", required=True)
    promote_parser.add_argument("--state-path", required=True)
    promote_parser.set_defaults(handler=promote)

    rollback_parser = commands.add_parser("rollback")
    rollback_parser.add_argument("--hf-repository", required=True)
    rollback_parser.add_argument("--state-path", required=True)
    rollback_parser.set_defaults(handler=rollback)

    callback_parser = commands.add_parser("callback")
    callback_parser.add_argument(
        "--status", choices=("active", "failed", "rolled_back"), required=True
    )
    callback_parser.add_argument("--candidate-mlflow-run-id", required=True)
    callback_parser.add_argument("--evaluation-run-id", required=True)
    callback_parser.add_argument("--github-run-id", required=True)
    callback_parser.add_argument("--approved-by", required=True)
    callback_parser.add_argument("--approved-at", required=True)
    callback_parser.add_argument("--deployed-revision")
    callback_parser.add_argument("--failure-reason")
    callback_parser.set_defaults(handler=callback)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
