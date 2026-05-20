# Todo

- [x] Create mirrored docs structure
- [x] Seed starter documentation
- [x] Create ADR starter set
- [x] Create `tasks/lessons.md`
- [x] Create workspace hooks and skills
- [x] Create `src/` package tree
- [x] Extract config, data, model, and baseline training modules from notebook
- [x] Add focused tests for extracted modules
- [x] Stabilize the extracted training slice with `uv run ruff`, `uv run mypy`, and focused `uv run pytest`
- [x] Rewrite notebook cells as thin wrappers around `src/` modules
      Applied directly to `mlops` notebook (Cell 3: create_training_config, Cell 7: removed redundant load, Cells 13/14: config.mlflow_model_name).
- [ ] Decide whether to move `src/` to `ml/src` before expanding the module surface
- [x] Split dependencies by concern (`ml`, `mlops`, `dev`) in `pyproject.toml`
- [ ] Implement `mlops` orchestration on the `mlops` branch