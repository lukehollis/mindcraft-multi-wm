from __future__ import annotations

import json
import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from mindcraft.world_model import ActionConditionedWorldModel, latent_prediction_loss


@dataclass(frozen=True, slots=True)
class RoboticsSkill:
    name: str
    command: tuple[float, float, float]
    description: str


ROBOTICS_SKILLS: dict[str, RoboticsSkill] = {
    "stand": RoboticsSkill("stand", (0.0, 0.0, 0.0), "Hold posture with near-zero base velocity."),
    "walk_forward_slow": RoboticsSkill(
        "walk_forward_slow", (0.35, 0.0, 0.0), "Track a cautious forward base velocity."
    ),
    "walk_forward": RoboticsSkill("walk_forward", (0.65, 0.0, 0.0), "Track a forward base velocity."),
    "walk_backward": RoboticsSkill("walk_backward", (-0.35, 0.0, 0.0), "Track a backward base velocity."),
    "strafe_left": RoboticsSkill("strafe_left", (0.0, 0.35, 0.0), "Track a left lateral base velocity."),
    "strafe_right": RoboticsSkill("strafe_right", (0.0, -0.35, 0.0), "Track a right lateral base velocity."),
    "turn_left": RoboticsSkill("turn_left", (0.0, 0.0, 0.65), "Track a positive yaw velocity."),
    "turn_right": RoboticsSkill("turn_right", (0.0, 0.0, -0.65), "Track a negative yaw velocity."),
    "arc_left": RoboticsSkill("arc_left", (0.45, 0.0, 0.45), "Walk forward while turning left."),
    "arc_right": RoboticsSkill("arc_right", (0.45, 0.0, -0.45), "Walk forward while turning right."),
}
ROBOTICS_SKILL_NAMES = tuple(ROBOTICS_SKILLS)
ROBOTICS_ACTION_DIM = len(ROBOTICS_SKILL_NAMES)


@dataclass(frozen=True, slots=True)
class CooperativeGoalMetrics:
    centroid: tuple[float, float]
    goal_xy: tuple[float, float]
    centroid_distance: float
    slot_targets: list[tuple[float, float]]
    slot_errors: list[float]
    mean_slot_error: float
    max_slot_error: float
    achieved: bool

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "centroid": list(self.centroid),
            "goal_xy": list(self.goal_xy),
            "centroid_distance": self.centroid_distance,
            "slot_targets": [list(target) for target in self.slot_targets],
            "slot_errors": self.slot_errors,
            "mean_slot_error": self.mean_slot_error,
            "max_slot_error": self.max_slot_error,
            "achieved": self.achieved,
        }


def centered_formation_offsets(num_agents: int, spacing: float = 1.25) -> np.ndarray:
    """Return deterministic formation slots centered around a shared goal."""
    if num_agents <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    columns = int(math.ceil(math.sqrt(num_agents)))
    rows = int(math.ceil(num_agents / columns))
    offsets: list[tuple[float, float]] = []
    for index in range(num_agents):
        row = index // columns
        column = index % columns
        x = (column - 0.5 * (columns - 1)) * spacing
        y = (row - 0.5 * (rows - 1)) * spacing
        offsets.append((x, y))
    arr = np.asarray(offsets, dtype=np.float32)
    return arr - arr.mean(axis=0, keepdims=True)


def cooperative_goal_metrics(
    root_xy: np.ndarray | list[list[float]],
    goal_xy: np.ndarray | list[float],
    slot_offsets: np.ndarray | list[list[float]],
    *,
    goal_radius: float = 0.35,
    slot_tolerance: float = 0.55,
) -> CooperativeGoalMetrics:
    positions = np.asarray(root_xy, dtype=np.float32).reshape(-1, 2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    offsets = np.asarray(slot_offsets, dtype=np.float32).reshape(-1, 2)
    if offsets.shape[0] != positions.shape[0]:
        offsets = centered_formation_offsets(positions.shape[0], spacing=slot_tolerance * 2.0)
    targets = goal[None, :] + offsets
    centroid = positions.mean(axis=0) if positions.size else np.zeros(2, dtype=np.float32)
    slot_errors_arr = np.linalg.norm(positions - targets, axis=1) if positions.size else np.zeros(0, dtype=np.float32)
    centroid_distance = float(np.linalg.norm(centroid - goal))
    mean_slot_error = float(slot_errors_arr.mean()) if slot_errors_arr.size else 0.0
    max_slot_error = float(slot_errors_arr.max()) if slot_errors_arr.size else 0.0
    achieved = bool(centroid_distance <= goal_radius and mean_slot_error <= slot_tolerance)
    return CooperativeGoalMetrics(
        centroid=(float(centroid[0]), float(centroid[1])),
        goal_xy=(float(goal[0]), float(goal[1])),
        centroid_distance=centroid_distance,
        slot_targets=[(float(target[0]), float(target[1])) for target in targets],
        slot_errors=[float(error) for error in slot_errors_arr],
        mean_slot_error=mean_slot_error,
        max_slot_error=max_slot_error,
        achieved=achieved,
    )


def body_velocity_to_world_xy(command: tuple[float, float, float], yaw_rad: float) -> np.ndarray:
    vx, vy, _wz = command
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray([cos_yaw * vx - sin_yaw * vy, sin_yaw * vx + cos_yaw * vy], dtype=np.float32)


def cooperative_skill_score(
    env_id: int,
    skill: str,
    root_xy: np.ndarray | list[list[float]],
    root_yaw: np.ndarray | list[float],
    goal_xy: np.ndarray | list[float],
    slot_offsets: np.ndarray | list[list[float]],
    *,
    skill_horizon_s: float,
    goal_radius: float = 0.35,
    slot_tolerance: float = 0.55,
    centroid_weight: float = 1.25,
    slot_weight: float = 1.0,
) -> dict[str, float]:
    positions = np.asarray(root_xy, dtype=np.float32).reshape(-1, 2)
    yaws = np.asarray(root_yaw, dtype=np.float32).reshape(-1)
    before = cooperative_goal_metrics(
        positions,
        goal_xy,
        slot_offsets,
        goal_radius=goal_radius,
        slot_tolerance=slot_tolerance,
    )
    predicted = positions.copy()
    if 0 <= env_id < predicted.shape[0] and skill in ROBOTICS_SKILLS:
        yaw = float(yaws[env_id]) if env_id < yaws.shape[0] else 0.0
        predicted[env_id] += body_velocity_to_world_xy(ROBOTICS_SKILLS[skill].command, yaw) * float(skill_horizon_s)
    after = cooperative_goal_metrics(
        predicted,
        goal_xy,
        slot_offsets,
        goal_radius=goal_radius,
        slot_tolerance=slot_tolerance,
    )
    slot_before = before.slot_errors[env_id] if env_id < len(before.slot_errors) else before.mean_slot_error
    slot_after = after.slot_errors[env_id] if env_id < len(after.slot_errors) else after.mean_slot_error
    centroid_progress = before.centroid_distance - after.centroid_distance
    formation_progress = before.mean_slot_error - after.mean_slot_error
    slot_progress = slot_before - slot_after
    achievement_bonus = 1.0 if after.achieved and not before.achieved else 0.0
    score = (
        float(centroid_weight) * centroid_progress
        + float(slot_weight) * (0.5 * formation_progress + slot_progress)
        + achievement_bonus
    )
    return {
        "score": float(score),
        "centroid_progress": float(centroid_progress),
        "formation_progress": float(formation_progress),
        "slot_progress": float(slot_progress),
        "predicted_centroid_distance": after.centroid_distance,
        "predicted_mean_slot_error": after.mean_slot_error,
        "achievement_bonus": achievement_bonus,
    }


@dataclass(slots=True)
class RoboticsTransition:
    episode: int
    step: int
    env_id: int
    skill: str
    command: tuple[float, float, float]
    observation: list[float]
    reward: float
    next_observation: list[float]
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "env_id": self.env_id,
            "skill": self.skill,
            "command": list(self.command),
            "observation": list(self.observation),
            "reward": self.reward,
            "next_observation": list(self.next_observation),
            "done": self.done,
            "info": self.info,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "RoboticsTransition":
        command = data.get("command") or ROBOTICS_SKILLS[str(data["skill"])].command
        return cls(
            episode=int(data["episode"]),
            step=int(data["step"]),
            env_id=int(data.get("env_id", 0)),
            skill=str(data["skill"]),
            command=tuple(float(x) for x in command[:3]),  # type: ignore[arg-type]
            observation=[float(x) for x in data["observation"]],
            reward=float(data["reward"]),
            next_observation=[float(x) for x in data["next_observation"]],
            done=bool(data.get("done", False)),
            info=dict(data.get("info") or {}),
            timestamp=float(data.get("timestamp", time())),
        )


class RoboticsReplayBuffer:
    def __init__(self, path: Path, capacity: int = 100_000):
        self.path = path
        self.capacity = capacity
        self.items: deque[RoboticsTransition] = deque(maxlen=capacity)
        self._offset = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.refresh()

    def __len__(self) -> int:
        return len(self.items)

    def append(self, transition: RoboticsTransition) -> None:
        self.items.append(transition)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(transition.to_jsonable(), sort_keys=True) + "\n")
            self._offset = f.tell()

    def refresh(self) -> int:
        if not self.path.exists():
            return 0
        if self.path.stat().st_size < self._offset:
            self.items.clear()
            self._offset = 0
        return self._load_new()

    def tail(self, count: int) -> Iterable[RoboticsTransition]:
        return list(self.items)[-count:]

    def can_sample_sequence(self, sequence_length: int) -> bool:
        return bool(self._sequence_windows(sequence_length))

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        rng: random.Random,
        *,
        holdout: bool | None = False,
    ) -> list[list[RoboticsTransition]]:
        if batch_size <= 0 or sequence_length <= 0:
            return []
        windows = self._sequence_windows(sequence_length, holdout=holdout)
        if not windows and holdout is False:
            windows = self._sequence_windows(sequence_length, holdout=None)
        if not windows:
            return []
        buckets: dict[tuple[str, int, bool], list[list[RoboticsTransition]]] = defaultdict(list)
        for window in windows:
            last = window[-1]
            buckets[(last.skill, last.env_id, last.done)].append(window)
        keys = list(buckets)
        rng.shuffle(keys)
        chosen: list[list[RoboticsTransition]] = []
        for key in keys:
            chosen.append(list(rng.choice(buckets[key])))
            if len(chosen) >= batch_size:
                rng.shuffle(chosen)
                return chosen
        while len(chosen) < batch_size:
            key = rng.choice(keys)
            chosen.append(list(rng.choice(buckets[key])))
        rng.shuffle(chosen)
        return chosen

    def _load_new(self) -> int:
        loaded = 0
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._offset)
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(RoboticsTransition.from_jsonable(json.loads(line)))
                except JSONDecodeError:
                    self._offset = line_start
                    break
                loaded += 1
                self._offset = f.tell()
        return loaded

    def _sequence_windows(self, sequence_length: int, holdout: bool | None = None) -> list[list[RoboticsTransition]]:
        if len(self.items) < sequence_length:
            return []
        trajectories: dict[tuple[int, int], list[RoboticsTransition]] = defaultdict(list)
        for transition in self.items:
            trajectories[(transition.episode, transition.env_id)].append(transition)
        windows: list[list[RoboticsTransition]] = []
        window_index = 0
        for trajectory in trajectories.values():
            if len(trajectory) < sequence_length:
                continue
            max_start = len(trajectory) - sequence_length
            for start in range(max_start + 1):
                is_holdout = window_index % 5 == 4
                window_index += 1
                if holdout is None or holdout == is_holdout:
                    windows.append(trajectory[start : start + sequence_length])
        return windows


@dataclass(slots=True)
class RoboticsSkillStats:
    attempts: int = 0
    successes: int = 0
    total_reward: float = 0.0
    q_value: float = 0.0
    prediction_error: float = 1.0
    last_episode: int = -1
    last_step: int = -1
    notes: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)


class RoboticsSkillLibrary:
    def __init__(self, path: Path, alpha: float = 0.15):
        self.path = path
        self.alpha = alpha
        self.stats: dict[str, RoboticsSkillStats] = {name: RoboticsSkillStats() for name in ROBOTICS_SKILL_NAMES}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.load()

    @property
    def total_attempts(self) -> int:
        return sum(stat.attempts for stat in self.stats.values())

    def score_candidates(self, ucb_c: float = 0.7, curiosity_weight: float = 0.2) -> dict[str, float]:
        total = max(1, self.total_attempts)
        scores: dict[str, float] = {}
        for name in ROBOTICS_SKILL_NAMES:
            stat = self.stats[name]
            ucb = ucb_c * math.sqrt(math.log(total + 1) / (stat.attempts + 1))
            curiosity = curiosity_weight * stat.prediction_error
            scores[name] = stat.q_value + ucb + curiosity
        return scores

    def update(
        self,
        skill: str,
        reward: float,
        success: bool,
        episode: int,
        step: int,
        prediction_error: float | None = None,
        note: str | None = None,
    ) -> None:
        if skill not in self.stats:
            self.stats[skill] = RoboticsSkillStats()
        stat = self.stats[skill]
        stat.attempts += 1
        stat.successes += int(success)
        stat.total_reward += float(reward)
        stat.q_value = (1.0 - self.alpha) * stat.q_value + self.alpha * float(reward)
        if prediction_error is None:
            stat.prediction_error *= 0.995
        else:
            stat.prediction_error = 0.9 * stat.prediction_error + 0.1 * float(prediction_error)
        stat.last_episode = episode
        stat.last_step = step
        if note:
            stat.notes = (stat.notes + [note])[-12:]

    def save(self) -> None:
        payload = {
            "version": 1,
            "skills": {name: asdict(stat) for name, stat in sorted(self.stats.items())},
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        skills = payload.get("skills", payload)
        for name, stat in skills.items():
            if name in ROBOTICS_SKILLS:
                self.stats[name] = RoboticsSkillStats(**stat)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "attempts": stat.attempts,
                "success_rate": stat.success_rate,
                "q_value": stat.q_value,
                "prediction_error": stat.prediction_error,
            }
            for name, stat in self.stats.items()
        }


@dataclass(slots=True)
class RoboticsWorldModelMetrics:
    train_step: int
    loss: float
    obs_loss: float
    jepa_loss: float
    reward_loss: float
    value_loss: float
    policy_loss: float
    done_loss: float
    code_usage: float
    val_loss: float | None = None


class RoboticsWorldModelTrainer:
    def __init__(
        self,
        storage_dir: Path,
        obs_dim: int,
        *,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        latent_dim: int = 16,
        ssm_state_dim: int = 16,
        ensemble_size: int = 3,
        lora_rank: int = 4,
        freeze_base_for_lora: bool = False,
        lr: float = 3.0e-4,
        gamma: float = 0.97,
        device: str | None = None,
        checkpoint_name: str = "go2_world_model.pt",
    ):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.obs_dim = int(obs_dim)
        self.action_dim = ROBOTICS_ACTION_DIM
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = float(gamma)
        self.model_config = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "latent_dim": latent_dim,
            "ssm_state_dim": ssm_state_dim,
            "ensemble_size": ensemble_size,
            "lora_rank": lora_rank,
            "freeze_base_for_lora": freeze_base_for_lora,
        }
        self.model = ActionConditionedWorldModel(
            self.obs_dim,
            self.action_dim,
            d_model=d_model,
            layers=layers,
            heads=heads,
            latent_dim=latent_dim,
            ssm_state_dim=ssm_state_dim,
            ensemble_size=ensemble_size,
            lora_rank=lora_rank,
            freeze_base_for_lora=freeze_base_for_lora,
        ).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1.0e-4)
        self.last_loss = 1.0
        self.train_step = 0
        self.checkpoint_path = self.storage_dir / checkpoint_name
        self.checkpoint_meta_path = self.storage_dir / f"{self.checkpoint_path.stem}_checkpoint.json"
        if self.checkpoint_path.exists():
            try:
                self.load()
            except RuntimeError as exc:
                print(f"ignoring incompatible robotics world model checkpoint {self.checkpoint_path}: {exc}")

    def train_batches(
        self,
        sequences: list[list[RoboticsTransition]],
        validation_sequences: list[list[RoboticsTransition]] | None = None,
    ) -> RoboticsWorldModelMetrics | None:
        if not sequences:
            return None
        self.model.train()
        obs, action, next_obs, reward, done = robotics_batch_to_tensors(sequences, self.device, self.obs_dim)
        pred = self.model(obs, action, next_obs=next_obs)
        losses = self._loss_components(pred, action, next_obs, reward, done)
        loss = losses["loss"]
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optim.step()
        self.model.update_target_encoder(tau=0.01)
        code_usage = pred["codes"].unique().numel() / max(1, pred["codes"].numel())
        self.last_loss = float(loss.detach().cpu())
        self.train_step += 1
        validation = self.evaluate_batches(validation_sequences or [])
        return RoboticsWorldModelMetrics(
            train_step=self.train_step,
            loss=self.last_loss,
            obs_loss=float(losses["obs_loss"].detach().cpu()),
            jepa_loss=float(losses["jepa_loss"].detach().cpu()),
            reward_loss=float(losses["reward_loss"].detach().cpu()),
            value_loss=float(losses["value_loss"].detach().cpu()),
            policy_loss=float(losses["policy_loss"].detach().cpu()),
            done_loss=float(losses["done_loss"].detach().cpu()),
            code_usage=float(code_usage),
            **validation,
        )

    @torch.no_grad()
    def evaluate_batches(self, sequences: list[list[RoboticsTransition]]) -> dict[str, float]:
        if not sequences:
            return {}
        self.model.eval()
        obs, action, next_obs, reward, done = robotics_batch_to_tensors(sequences, self.device, self.obs_dim)
        pred = self.model(obs, action, next_obs=next_obs)
        losses = self._loss_components(pred, action, next_obs, reward, done)
        return {"val_loss": float(losses["loss"].detach().cpu())}

    def _loss_components(
        self,
        pred: dict[str, torch.Tensor],
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        returns = robotics_discounted_returns(reward, done, self.gamma)
        policy_target = action.argmax(dim=-1)
        obs_loss = F.smooth_l1_loss(pred["next_obs"], next_obs)
        jepa_loss = latent_prediction_loss(pred["pred_latent"], pred["target_latent"])
        reward_target = reward.unsqueeze(-1).expand_as(pred["reward_members"])
        reward_loss = F.mse_loss(pred["reward_members"], reward_target)
        value_target = returns.unsqueeze(-1).expand_as(pred["value_members"])
        value_loss = F.smooth_l1_loss(pred["value_members"], value_target)
        policy_loss = F.cross_entropy(pred["policy"].reshape(-1, self.action_dim), policy_target.reshape(-1))
        done_loss = F.binary_cross_entropy_with_logits(pred["done"], done)
        loss = obs_loss + jepa_loss + reward_loss + 0.5 * value_loss + 0.2 * policy_loss + 0.2 * done_loss
        return {
            "loss": loss,
            "obs_loss": obs_loss,
            "jepa_loss": jepa_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "done_loss": done_loss,
        }

    @torch.no_grad()
    def prediction_error(self, transition: RoboticsTransition) -> float:
        self.model.eval()
        obs, action, next_obs, reward, _done = robotics_batch_to_tensors([[transition]], self.device, self.obs_dim)
        pred = self.model(obs, action)
        obs_err = F.smooth_l1_loss(pred["next_obs"], next_obs).item()
        reward_err = F.mse_loss(pred["reward"], reward).item()
        return float(obs_err + reward_err)

    @torch.no_grad()
    def predict_skill(self, observation: np.ndarray | list[float], skill: str) -> dict[str, Any]:
        self.model.eval()
        obs_array = coerce_observation(observation, self.obs_dim)
        obs = torch.tensor(obs_array, dtype=torch.float32, device=self.device).view(1, 1, -1)
        action = torch.tensor(encode_robotics_action(skill), dtype=torch.float32, device=self.device).view(1, 1, -1)
        pred = self.model(obs, action)
        policy = torch.softmax(pred["policy"][0, -1], dim=-1)
        index = ROBOTICS_SKILL_NAMES.index(skill)
        return {
            "next_obs": pred["next_obs"][0, -1].detach().cpu().numpy(),
            "reward": float(pred["reward"][0, -1].detach().cpu()),
            "reward_uncertainty": float(pred["reward_uncertainty"][0, -1].detach().cpu()),
            "value": float(pred["value"][0, -1].detach().cpu()),
            "value_uncertainty": float(pred["value_uncertainty"][0, -1].detach().cpu()),
            "model_uncertainty": float(
                (pred["reward_uncertainty"][0, -1] + self.gamma * pred["value_uncertainty"][0, -1]).detach().cpu()
            ),
            "done_logit": float(pred["done"][0, -1].detach().cpu()),
            "prior": float(policy[index].detach().cpu()),
        }

    def save(self) -> None:
        torch.save(
            {
                "checkpoint_version": 1,
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "last_loss": self.last_loss,
                "train_step": self.train_step,
                "gamma": self.gamma,
                "skill_names": ROBOTICS_SKILL_NAMES,
                "model_config": self.model_config,
            },
            self.checkpoint_path,
        )
        self.checkpoint_meta_path.write_text(
            json.dumps(self.checkpoint_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self) -> None:
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        if tuple(payload.get("skill_names") or ()) != ROBOTICS_SKILL_NAMES:
            raise RuntimeError("checkpoint robotics skill vocabulary does not match current model")
        if dict(payload.get("model_config") or {}) != self.model_config:
            raise RuntimeError("checkpoint model config does not match current robotics model")
        self.model.load_state_dict(payload["model"])
        if "optim" in payload:
            self.optim.load_state_dict(payload["optim"])
        self.last_loss = float(payload.get("last_loss", 1.0))
        self.train_step = int(payload.get("train_step", 0))
        self.gamma = float(payload.get("gamma", self.gamma))

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "checkpoint_path": str(self.checkpoint_path),
            "last_loss": self.last_loss,
            "train_step": self.train_step,
            "gamma": self.gamma,
            "skill_count": len(ROBOTICS_SKILL_NAMES),
            "skill_names": list(ROBOTICS_SKILL_NAMES),
            "model_config": self.model_config,
        }


def choose_robotics_skill(
    observation: np.ndarray | list[float],
    library: RoboticsSkillLibrary,
    rng: random.Random,
    *,
    world_model: RoboticsWorldModelTrainer | None = None,
    replay_size: int = 0,
    epsilon: float = 0.12,
    model_min_replay: int = 64,
    ucb_c: float = 0.7,
    curiosity_weight: float = 0.2,
    uncertainty_weight: float = 0.35,
) -> tuple[str, str, dict[str, Any]]:
    if rng.random() < epsilon:
        return rng.choice(ROBOTICS_SKILL_NAMES), "epsilon", {}
    library_scores = library.score_candidates(ucb_c=ucb_c, curiosity_weight=curiosity_weight)
    if world_model is None or replay_size < model_min_replay:
        skill = max(ROBOTICS_SKILL_NAMES, key=lambda name: (library_scores.get(name, float("-inf")), name))
        return skill, "skill_library", {}
    diagnostics: dict[str, Any] = {}
    best_skill = ROBOTICS_SKILL_NAMES[0]
    best_score = float("-inf")
    for skill in ROBOTICS_SKILL_NAMES:
        prediction = world_model.predict_skill(observation, skill)
        score = (
            library_scores.get(skill, 0.0)
            + prediction["reward"]
            + world_model.gamma * prediction["value"]
            + 0.1 * prediction["prior"]
            - uncertainty_weight * prediction["model_uncertainty"]
        )
        diagnostics[skill] = {
            "score": score,
            "library_score": library_scores.get(skill, 0.0),
            "reward": prediction["reward"],
            "value": prediction["value"],
            "uncertainty": prediction["model_uncertainty"],
            "prior": prediction["prior"],
        }
        if score > best_score:
            best_score = score
            best_skill = skill
    return best_skill, "robotics_world_model", diagnostics


def encode_robotics_action(skill: str) -> np.ndarray:
    arr = np.zeros(ROBOTICS_ACTION_DIM, dtype=np.float32)
    if skill in ROBOTICS_SKILL_NAMES:
        arr[ROBOTICS_SKILL_NAMES.index(skill)] = 1.0
    return arr


def coerce_observation(observation: np.ndarray | list[float], obs_dim: int) -> np.ndarray:
    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    if arr.shape[0] == obs_dim:
        return arr
    if arr.shape[0] > obs_dim:
        return arr[:obs_dim]
    padded = np.zeros(obs_dim, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def robotics_batch_to_tensors(
    sequences: list[list[RoboticsTransition]],
    device: torch.device,
    obs_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_rows: list[list[np.ndarray]] = []
    action_rows: list[list[np.ndarray]] = []
    next_rows: list[list[np.ndarray]] = []
    reward_rows: list[list[float]] = []
    done_rows: list[list[float]] = []
    for sequence in sequences:
        obs_rows.append([coerce_observation(t.observation, obs_dim) for t in sequence])
        action_rows.append([encode_robotics_action(t.skill) for t in sequence])
        next_rows.append([coerce_observation(t.next_observation, obs_dim) for t in sequence])
        reward_rows.append([float(t.reward) for t in sequence])
        done_rows.append([float(t.done) for t in sequence])
    return (
        torch.tensor(np.asarray(obs_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(action_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(next_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(reward_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(done_rows), dtype=torch.float32, device=device),
    )


def robotics_discounted_returns(reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[0], device=reward.device, dtype=reward.dtype)
    for t in range(reward.shape[1] - 1, -1, -1):
        running = reward[:, t] + gamma * running * (1.0 - done[:, t])
        returns[:, t] = running
    return returns
