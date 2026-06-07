from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MinecraftConfig:
    bridge_uri: str = "ws://localhost:8765"
    host: str = "localhost"
    port: int = 25565
    version: str | None = None
    bot_names: list[str] = field(default_factory=lambda: ["agent_0", "agent_1"])
    action_timeout_s: float = 90.0


@dataclass(slots=True)
class LearningConfig:
    seed: int = 7
    epsilon: float = 0.12
    ucb_c: float = 0.7
    curiosity_weight: float = 0.25
    gamma: float = 0.97
    replay_capacity: int = 50_000
    frontier_sampling: bool = True
    hindsight_relabeling: bool = True
    sequence_length: int = 8
    batch_size: int = 16
    train_every_steps: int = 4
    train_batches: int = 1
    warmup_transitions: int = 16
    model_dim: int = 128
    model_layers: int = 3
    model_heads: int = 4
    latent_dim: int = 16
    ssm_state_dim: int = 16
    world_model_ensemble: int = 3
    lora_rank: int = 4
    freeze_base_for_lora: bool = False
    mcts_simulations: int = 32
    mcts_depth: int = 3
    mcts_c_puct: float = 1.35
    mcts_uncertainty_weight: float = 0.35
    mcts_affordance_gate: float = 5.0
    mcts_min_replay: int = 1_000
    lr: float = 3.0e-4
    checkpoint_every_steps: int = 50
    checkpoint_name: str = "world_model.pt"
    reload_checkpoint_path: Path | None = None
    reload_checkpoint_every_steps: int = 0


@dataclass(slots=True)
class TelemetryConfig:
    enabled: bool = True
    weave_project: str = "mindcraft/minecraft-agent"
    wandb_project: str = "mindcraft"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
    run_name: str = "mindcraft"
    tensorboard_enabled: bool = True
    tensorboard_dir: str = "tensorboard"


@dataclass(slots=True)
class RunConfig:
    storage_dir: Path = Path("runs/default")
    episodes: int = 3
    steps_per_episode: int = 100
    duration_s: float | None = None
    bridge: str = "live"
    rally_point: tuple[int, int, int] | None = None


@dataclass(slots=True)
class MindcraftConfig:
    minecraft: MinecraftConfig = field(default_factory=MinecraftConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> MindcraftConfig:
    data: dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        data = loaded
    if overrides:
        data = _merge(data, overrides)

    cfg = MindcraftConfig()
    if "minecraft" in data:
        cfg.minecraft = MinecraftConfig(**data["minecraft"])
    if "learning" in data:
        learning_data = dict(data["learning"])
        if learning_data.get("reload_checkpoint_path") is not None:
            learning_data["reload_checkpoint_path"] = _coerce_path(learning_data["reload_checkpoint_path"])
        cfg.learning = LearningConfig(**learning_data)
    if "telemetry" in data:
        cfg.telemetry = TelemetryConfig(**data["telemetry"])
    if "run" in data:
        run_data = dict(data["run"])
        if "storage_dir" in run_data:
            run_data["storage_dir"] = _coerce_path(run_data["storage_dir"])
        if run_data.get("rally_point") is not None:
            run_data["rally_point"] = tuple(run_data["rally_point"])
        cfg.run = RunConfig(**run_data)
    return cfg
