# AGENTS.md

## Start Here

At the start of every session, read these files before making changes:
- `tasks/todo.md`
- `tasks/lessons.md`
- `tasks/session-updates.md`
- `tasks/agile.md`

Use them to get up to speed on current progress, prior mistakes, and the next planned slice of work.

## Repo Role

`sicurre-ml` owns the ML training pipeline for Sicurre.

This repo does:
- extract notebook logic into reusable `src/` modules
- train, evaluate, and track experiments
- publish approved model artifacts
- host MLOps automation on the `mlops` branch

This repo does not:
- own the operational data platform
- connect to an app database
- run the user-facing application runtime

## Boundaries

- `sicurre` owns ingestion, normalization, dataset export, lineage, and app runtime.
- Cloudflare R2 is the source of truth for frozen training datasets.
- Kaggle is the execution runtime for training, not the canonical data store.
- Hugging Face is the model delivery boundary back to the app.

## Branch Discipline

- `ml`: notebook-to-module extraction, training code (`src/config`, `src/data`, `src/model`, `src/training`), and tests
- `mlops`: evaluation, MLflow logging, promotion flows, Kaggle triggers, dataset sync, HF publication

## Working Rules

- Documentation and setup scaffolding come before code implementation.
- Keep notebook cells thin; move reusable logic into `src/` modules.
- Ignore the notebook's last inference/demo cell during `ml` branch extraction.
- Keep changes minimal and aligned with the documented contracts in `docs/`.
- Keep tests organized only under `tests/unit`, `tests/integration`, and `tests/e2e`.
- Keep hook definitions in `.github/hooks/` and hook implementations in `.github/scripts/hooks/`.
- Treat `scripts/hooks/` as compatibility shims only; do not put real hook logic there.
- ml/ and mlops/ at repo root are notebook and operations containers, not Python packages. All importable Python code lives under src/.

## Secret Hygiene

- No script, module, or notebook cell may hardcode a secret value (token, password, API key, credential).
- All secrets must be loaded via load_secrets() in src/config/training_config.py, which uses platform-appropriate backends: Kaggle UserSecretsClient, Colab userdata, or local .env via python-dotenv.
- Any value that is environment-specific or user-configurable (model name, dataset path, experiment name) must be parameterized through TrainingConfig, RuntimeState, or a .env entry — never hardcoded in source.
- The stop hook scans for common secret literal patterns and blocks the session if any are found.

## Validation

Prefer these checks after changes:

```bash
uv sync
uv run ruff check src .github/scripts tests
uv run mypy src .github/scripts tests
uv run pytest
uv run python -m compileall src .github/scripts scripts tests
```

If zero-error checks fail, resolve them before ending the session.

## References

- `docs/model/training-plan.md`
- `docs/architecture/sync-contracts.md`
- `docs/ops/kaggle-runbook.md`