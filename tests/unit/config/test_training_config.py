from src.config.training_config import create_training_config, load_secrets


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
