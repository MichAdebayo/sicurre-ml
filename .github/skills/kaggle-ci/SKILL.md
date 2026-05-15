---
name: kaggle-ci
description: "Trigger Kaggle dataset and notebook runs from CI. Use when: wiring GitHub Actions, pushing a Kaggle dataset version, syncing a notebook to Kaggle, or debugging Kaggle training automation."
---

# Kaggle CI

## When to Use

- preparing the mlops branch for dataset sync and training automation
- debugging kaggle datasets version or kaggle kernels push
- documenting the bridge from R2-backed frozen datasets to Kaggle execution

## Contract Reminder

- R2 remains the canonical frozen dataset store
- Kaggle Dataset is the packaged execution mirror
- the notebook source stays in-repo; a copy is pushed to Kaggle for execution

## Typical Commands

```bash
kaggle datasets version --dir path/to/frozen-export --message "Dataset v${DATASET_VERSION}" --dir-mode zip
kaggle kernels push --path .
```

## Current Branch Rule

Do not implement orchestration on the ml branch. CI automation belongs on mlops.