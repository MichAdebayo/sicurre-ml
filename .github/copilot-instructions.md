# Copilot Instructions

Read AGENTS.md first at the start of each session. Then review the current task state in:
- tasks/todo.md
- tasks/lessons.md
- tasks/session-updates.md
- tasks/agile.md

## Scope

This repo owns ML training concerns only: modularized training code, evaluation, experiment tracking, and model publication.

Do not add:
- database access
- web API runtime code
- direct coupling to the companion app internals

## Architecture Rules

- Treat Cloudflare R2 as the canonical frozen dataset store.
- Treat Kaggle as the training execution environment.
- Keep notebook cells as thin wrappers around src modules.
- On the ml branch, focus only on reusable training code and tests.
- Do not implement the notebook's last inference/demo cell as part of the ml extraction slice.

## Session Workflow

- Setup and documentation come before code changes.
- Update task files as work progresses.
- Aim for zero diagnostics before ending a session.
- Keep tests organized only under tests/unit, tests/integration, and tests/e2e.
- Keep hook definitions in .github/hooks/ and the real hook scripts in .github/scripts/hooks/.
- Treat scripts/hooks/ as compatibility-only wrappers when legacy callers still expect that path.
- If errors remain, run uv run ruff check src .github/scripts tests and uv run mypy src .github/scripts tests, then use the repo's zero-error repair workflow before stopping.

## Secret Hygiene

- No script, module, or notebook cell may hardcode a secret value (token, password, API key, credential).
- All secrets must be loaded via load_secrets() in src/config/training_config.py, which uses platform-appropriate backends: Kaggle UserSecretsClient, Colab userdata, or local .env via python-dotenv.
- Any value that is environment-specific or user-configurable (model name, dataset path, experiment name) must be parameterized through TrainingConfig, RuntimeState, or a .env entry — never hardcoded in source.
- The stop hook scans for common secret literal patterns and blocks the session if any are found.

## References

- docs/model/training-plan.md
- docs/architecture/component-design.md
- docs/architecture/sync-contracts.md