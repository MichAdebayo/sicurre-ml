from src.config.training_config import create_training_config
from src.training.args import compute_warmup_steps, create_training_args


def test_compute_warmup_steps_uses_ten_percent() -> None:
    assert compute_warmup_steps(train_size=100, batch_size=10, num_epochs=5) == 5


def test_create_training_args_sets_core_fields(tmp_path) -> None:
    config = create_training_config("cpu")
    args = create_training_args(
        config=config,
        run_name="baseline-vtest",
        output_dir=tmp_path,
        train_size=100,
    )
    assert args.run_name == "baseline-vtest"
    assert args.output_dir == str(tmp_path)
