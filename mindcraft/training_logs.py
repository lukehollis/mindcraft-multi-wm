from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class TensorboardLogger:
    def __init__(self, storage_dir: Path, enabled: bool = True, log_dir: str | Path = "tensorboard"):
        self.enabled = enabled
        self.log_dir = _resolve_log_dir(storage_dir, log_dir)
        self.writer: Any | None = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            print(f"tensorboard unavailable; install tensorboard to enable local event logs: {exc}")
            self.enabled = False
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_agent_step(
        self,
        *,
        agent: str,
        reward: float,
        success: bool,
        duration_s: float,
        replay_size: int,
        env_step: int,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("agent/reward", reward, env_step)
        self.writer.add_scalar("agent/success", float(success), env_step)
        self.writer.add_scalar("agent/duration_s", duration_s, env_step)
        self.writer.add_scalar("replay/size", replay_size, env_step)
        self.writer.add_scalar(f"agents/{agent}/reward", reward, env_step)
        self.writer.add_scalar(f"agents/{agent}/success", float(success), env_step)

    def log_world_model(
        self,
        *,
        metrics: Any,
        replay_size: int,
        phase: str,
        env_step: int | None = None,
    ) -> None:
        if self.writer is None:
            return
        payload = asdict(metrics)
        train_step = int(payload["train_step"])
        for key, value in payload.items():
            if key == "train_step" or value is None:
                continue
            self.writer.add_scalar(f"world_model/{key}", float(value), train_step)
            self.writer.add_scalar(f"{phase}/world_model/{key}", float(value), train_step)
        self.writer.add_scalar("world_model/train_step", train_step, train_step)
        self.writer.add_scalar("replay/size_at_train", replay_size, train_step)
        if env_step is not None:
            self.writer.add_scalar("runtime/env_step_at_train", env_step, train_step)

    def log_progress(
        self,
        *,
        agent: str,
        skill: str,
        reward: float,
        success: bool,
        progress: dict[str, Any],
        env_step: int,
    ) -> None:
        if self.writer is None:
            return
        stage_index = float(progress.get("stage_index_after", 0))
        stage_delta = float(progress.get("stage_delta", 0))
        first_milestones = progress.get("first_milestones") or []
        self.writer.add_scalar("progress/stage_index", stage_index, env_step)
        self.writer.add_scalar("progress/stage_delta", stage_delta, env_step)
        self.writer.add_scalar("progress/milestone_first_gain", float(bool(first_milestones)), env_step)
        self.writer.add_scalar(f"progress_by_agent/{agent}/stage_index", stage_index, env_step)
        self.writer.add_scalar(f"progress_by_skill/{skill}/reward", reward, env_step)
        self.writer.add_scalar(f"progress_by_skill/{skill}/success", float(success), env_step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def append_training_metrics(
    storage_dir: Path,
    *,
    metrics: Any,
    replay_size: int,
    phase: str,
    env_step: int | None = None,
) -> None:
    payload = {
        "phase": phase,
        "replay_size": replay_size,
        **asdict(metrics),
    }
    if env_step is not None:
        payload["env_step"] = env_step
    path = storage_dir / "training_metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def append_progress_metrics(
    storage_dir: Path,
    *,
    agent: str,
    role: str,
    goal: str,
    skill: str,
    reward: float,
    success: bool,
    progress: dict[str, Any],
    env_step: int,
    episode: int,
    step: int,
) -> None:
    payload = {
        "agent": agent,
        "role": role,
        "goal": goal,
        "skill": skill,
        "reward": reward,
        "success": success,
        "env_step": env_step,
        "episode": episode,
        "step": step,
        **progress,
    }
    path = storage_dir / "progress_metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _resolve_log_dir(storage_dir: Path, log_dir: str | Path) -> Path:
    path = Path(log_dir)
    if path.is_absolute():
        return path
    return storage_dir / path
