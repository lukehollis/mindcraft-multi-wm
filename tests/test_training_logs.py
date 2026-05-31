from pathlib import Path

from mindcraft.world_model import WorldModelMetrics
from mindcraft.training_logs import TensorboardLogger, append_progress_metrics, append_training_metrics


def make_metrics(train_step: int = 1) -> WorldModelMetrics:
    return WorldModelMetrics(
        train_step=train_step,
        loss=1.0,
        obs_loss=0.2,
        jepa_loss=0.1,
        reward_loss=0.3,
        value_loss=0.4,
        policy_loss=0.5,
        done_loss=0.6,
        code_usage=0.7,
    )


def test_training_metrics_jsonl_and_tensorboard_events(tmp_path: Path) -> None:
    metrics = make_metrics()

    append_training_metrics(tmp_path, metrics=metrics, replay_size=12, phase="replay", env_step=8)
    logger = TensorboardLogger(tmp_path, enabled=True, log_dir="tensorboard")
    logger.log_world_model(metrics=metrics, replay_size=12, phase="replay", env_step=8)
    logger.log_agent_step(agent="agent_0", reward=1.25, success=True, duration_s=0.5, replay_size=12, env_step=8)
    logger.close()

    text = (tmp_path / "training_metrics.jsonl").read_text(encoding="utf-8")
    assert '"phase": "replay"' in text
    assert '"train_step": 1' in text
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


def test_progress_metrics_jsonl(tmp_path: Path) -> None:
    append_progress_metrics(
        tmp_path,
        agent="agent_0",
        role="agent",
        goal="tools",
        skill="craft_crafting_table",
        reward=6.0,
        success=True,
        progress={"stage_before": "wood", "stage_after": "table_setup", "stage_delta": 1},
        env_step=4,
        episode=0,
        step=2,
    )

    text = (tmp_path / "progress_metrics.jsonl").read_text(encoding="utf-8")
    assert '"skill": "craft_crafting_table"' in text
    assert '"stage_after": "table_setup"' in text
