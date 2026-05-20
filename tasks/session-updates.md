# Session Updates

This file is appended automatically by the `PreCompact` hook to preserve session progress before context compaction.

Review the latest entry here when resuming a session.

## 2026-05-15 17:09Z

- Completed the zero-error repair pass for the extracted training slice.
- `uv run ruff check src .github/scripts tests` passed.
- `uv run mypy src .github/scripts tests` passed.
- Focused test slice passed: 10 tests green.
- Confirmed hook scripts live under `.github/scripts/hooks`; `scripts/hooks/` is empty.
- Next decision pending: keep root `src/` or move to `ml/src` before further extraction.

## 2026-05-15 17:31Z

- Reorganized tests so `tests/` now contains only `unit/`, `integration/`, and `e2e/`.
- Moved all current Python tests under `tests/unit/`.
- Updated AGENTS, Copilot instructions, the zero-error remediator agent, and the stop hook to use canonical `.github/scripts` validation paths.
- Restored legacy `scripts/hooks/` wrappers only as compatibility shims for older callers.
- Verified `uv run ruff check src .github/scripts tests`, `uv run mypy src .github/scripts tests`, `uv run pytest`, and the legacy `scripts/hooks/stop_zero_errors.py` path all pass.

## 2026-05-20

- Diagnosed and fixed Kaggle training pipeline GPU crash (torch 2.10.0+cu128 incompatible with P100 SM_60; CUDA minimum raised to SM_70 in torch 2.6).
- Implemented nvidia-smi capability probe in notebook setup cell (runs before any `torch.cuda.*` call) to detect SM version without initializing the CUDA context, then conditionally downgrades to `torch==2.5.1+cu121` + matching `torchvision==0.20.1+cu121` + `torchaudio==2.5.1+cu121` for Pascal GPUs.
- Added `"machine_shape": "T4 x2"` to the CI-generated `kernel-metadata.json` in `.github/workflows/train.yml` to request T4 via the Kaggle kernels API (not yet confirmed — T4 enforcement pending next CI run).
- Confirmed training running end-to-end on P100 (version 19, manually pushed): all previous crash points cleared, warmup steps logged, training in progress.
- Split `pyproject.toml` dependencies into `ml`, `mlops`, and `dev` concern groups.
- Noted: `ml`-branch notebook refactoring is safe to push without triggering CI; only merging `ml/kaggle_training.ipynb` back to `mlops` would queue a new Kaggle run.

## 2026-05-21

### Branch reconciliation — making ml and mlops src/ consistent

- `ml` branch: replaced `src/evaluation/__init__.py` and `src/registry/__init__.py` with the canonical `mlops` implementations.
  - `evaluate_on_test` now uses `trainer.evaluate()` (not `trainer.predict()`) and removes `EarlyStoppingCallback` before eval.
  - `register_model` uses `mlflow.transformers.log_model()` with full pip_requirements.
  - `promote_if_threshold` uses `search_model_versions` and deletes @production alias before reassigning.
  - `push_to_hub` loads model/tokenizer from disk and calls `.push_to_hub()` directly.
  - Zero ruff + mypy errors on `ml`. Pushed as `db3d8f7`.

- `mlops` branch: applied `training_config.py` improvements from `ml`:
  - Added `mlflow_model_name: str = "main.sicurre.phishing-detector"` to `TrainingConfig`.
  - Added `data_dir: Path | None = None` and `output_dir: Path | None = None` overrides to `build_runtime_state()`.

- `mlops` notebook updated (6 cells):
  - **Cell 1** (`#VSC-e23c5378`): added `build_runtime_state`, `create_training_config` to imports.
  - **Cell 3** (`#VSC-cd34d7e7`): switched `TrainingConfig()` → `create_training_config(device)` — fixes silent batch_size=8 bug on CUDA (now batch_size=16 + fp16=True). Replaced manual `RuntimeState(...)` construction with `build_runtime_state(runtime_env=..., data_dir=..., output_dir=...)`. Replaced hardcoded `MODEL_NAME` with `config.mlflow_model_name`.
  - **Cell 7** (`#VSC-141a11a1`): removed redundant `load_model()` + `compute_class_weights()` — both are called internally by `prepare_baseline_training()`. Was wasting ~2 GB memory per run.
  - **Cell 8** (`#VSC-e51c9155`): added `print(f"Class weights: {setup.class_weights.tolist()}")`.
  - **Cells 13/14** (`#VSC-aac80fc7`, `#VSC-945eaf3a`): `MODEL_NAME` → `config.mlflow_model_name`.

- Zero ruff + mypy errors on `mlops`. Pushed as `94d8d7c`.
- NOTE: the `94d8d7c` push modified `src/config/training_config.py` and `ml/kaggle_training.ipynb` — both paths watched by `train.yml` — so a new Kaggle training run will be queued. This is intentional: the batch_size fix means the first properly-configured GPU run.
- `tasks/todo.md` updated: "Rewrite notebook cells" marked done.