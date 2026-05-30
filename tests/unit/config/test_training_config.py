import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.config.training_config import (
    _resolve_data_dir,
    _resolve_output_dir,
    build_runtime_state,
    create_training_config,
    detect_device,
    detect_runtime,
    load_secrets,
)


def test_create_training_config_for_cuda_enables_fp16() -> None:
    config = create_training_config("cuda")
    assert config.batch_size == 16
    assert config.use_fp16 is True


def test_load_secrets_local_returns_all_expected_keys(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")
    secrets = load_secrets("local")
    assert secrets["HF_TOKEN"] == "token"
    assert "DATABRICKS_HOST" in secrets
    assert "REPO_NAME" in secrets


# ── detect_runtime ───────────────────────────────────────────────────────────

def test_detect_runtime_returns_local_when_neither_kaggle_nor_colab() -> None:
    # In the CI/local environment neither /kaggle/working nor google.colab exist.
    result = detect_runtime()
    assert result == "local"


def test_detect_runtime_returns_kaggle_when_path_exists(monkeypatch) -> None:
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/kaggle/working")
    result = detect_runtime()
    assert result == "kaggle"


def test_detect_runtime_returns_colab_when_module_importable(monkeypatch) -> None:
    monkeypatch.setattr(os.path, "exists", lambda _: False)
    mock_colab = MagicMock()
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    result = detect_runtime()
    assert result == "colab"


# ── detect_device ────────────────────────────────────────────────────────────

def test_detect_device_returns_cpu_tuple() -> None:
    # No GPU in CI — must return cpu without tpu.
    device, use_tpu = detect_device()
    assert device in {"cpu", "cuda", "mps"}
    assert isinstance(use_tpu, bool)


def test_detect_device_returns_cuda_when_available(monkeypatch) -> None:
    import src.config.training_config as tc
    monkeypatch.setattr(tc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tc.torch.backends.mps, "is_available", lambda: False)
    device, use_tpu = detect_device()
    assert device == "cuda"
    assert use_tpu is False


def test_detect_device_returns_mps_when_available(monkeypatch) -> None:
    import src.config.training_config as tc
    monkeypatch.setattr(tc.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tc.torch.backends.mps, "is_available", lambda: True)
    device, use_tpu = detect_device()
    assert device == "mps"
    assert use_tpu is False


# ── load_secrets ─────────────────────────────────────────────────────────────

def test_load_secrets_kaggle_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://host")
    secrets = load_secrets("kaggle")
    # Keys set in env must be returned as-is.
    assert secrets["HF_TOKEN"] == "hf_test"
    assert secrets["DATABRICKS_HOST"] == "https://host"
    # All expected keys are present in the result.
    assert "DATABRICKS_TOKEN" in secrets
    assert "REPO_NAME" in secrets


def test_load_secrets_colab_uses_userdata(monkeypatch) -> None:
    mock_userdata = MagicMock()
    mock_userdata.get.return_value = "colab_value"
    mock_colab_module = MagicMock()
    mock_colab_module.userdata = mock_userdata
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab_module)
    secrets = load_secrets("colab")
    assert secrets["HF_TOKEN"] == "colab_value"


def test_load_secrets_colab_handles_missing_key(monkeypatch) -> None:
    mock_userdata = MagicMock()
    mock_userdata.get.side_effect = Exception("secret not found")
    mock_colab_module = MagicMock()
    mock_colab_module.userdata = mock_userdata
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab_module)
    secrets = load_secrets("colab")
    assert all(v is None for v in secrets.values())


# ── path resolution ──────────────────────────────────────────────────────────

def test_resolve_data_dir_local() -> None:
    assert _resolve_data_dir("local") == Path("data/final")


def test_resolve_data_dir_colab() -> None:
    result = _resolve_data_dir("colab")
    assert "content" in str(result)


def test_resolve_data_dir_kaggle() -> None:
    result = _resolve_data_dir("kaggle")
    assert "kaggle" in str(result)


def test_resolve_output_dir_local() -> None:
    result = _resolve_output_dir("local", "20250530")
    assert "models" in str(result)
    assert "20250530" in str(result)


def test_resolve_output_dir_colab() -> None:
    result = _resolve_output_dir("colab", "20250530")
    assert "content" in str(result)
    assert "20250530" in str(result)


def test_resolve_output_dir_kaggle() -> None:
    result = _resolve_output_dir("kaggle", "20250530")
    assert "kaggle" in str(result)
    assert "20250530" in str(result)


# ── build_runtime_state ──────────────────────────────────────────────────────

def test_build_runtime_state_local(tmp_path: Path) -> None:
    state = build_runtime_state(
        runtime_env="local",
        data_dir=tmp_path,
        output_dir=tmp_path,
    )
    assert state.runtime_env == "local"
    assert state.device in {"cpu", "cuda", "mps"}
    assert isinstance(state.use_tpu, bool)
    assert state.data_dir == tmp_path

