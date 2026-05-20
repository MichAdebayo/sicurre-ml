# Lessons Learned

## Repository workflow
- Documentation and setup scaffolding are a hard prerequisite before code implementation in this repo.
- Mirror the companion `sicurre` docs structure so architecture, ADRs, and runbooks are easy to compare during certification review.
- Use this file to capture lessons from repeated failures or from discovering a clearly better path, so handoff to another agent stays smooth.

## Kaggle GPU / PyTorch compatibility
- `torch.cuda.is_available()` initializes the CUDA context; env vars set after this call (e.g. `CUDA_VISIBLE_DEVICES`) have no effect. Use `nvidia-smi` via subprocess to detect GPU capability before any torch import.
- PyTorch dropped Pascal (SM_60/P100) support in torch 2.6.0. Last version supporting SM_60 is `torch==2.5.1+cu121`.
- When downgrading torch, always downgrade `torchvision` and `torchaudio` to the matching build. A version mismatch causes `RuntimeError: operator torchvision::nms does not exist`, which cascades into `ModuleNotFoundError` for `transformers.Trainer`.
- `kernel-metadata.json` (written by `kaggle kernels push`) overrides all embedded `{"kaggle": ...}` metadata in the notebook JSON. For API-triggered runs, configure accelerator/machine_shape in `kernel-metadata.json` only.

## CI branch safety
- The `train.yml` workflow fires only on pushes to `mlops` touching `src/**`, `ml/kaggle_training.ipynb`, or `.github/workflows/train.yml`. Pushing notebook refactoring to the `ml` branch is completely safe.
- Merging an updated notebook from `ml` back to `mlops` will touch `ml/kaggle_training.ipynb` and queue a new Kaggle run. Coordinate merge timing with training schedule.
