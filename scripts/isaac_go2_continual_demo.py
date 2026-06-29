#!/usr/bin/env python3
from __future__ import annotations

"""Continual-learning Go2 demo for Isaac Lab.

Run this through Isaac Lab's Python launcher, for example:

    ./isaaclab.sh -p /path/to/mindcraft/scripts/isaac_go2_continual_demo.py --headless
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the Mindcraft continual world model on Unitree Go2 in Isaac Lab.")
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--storage-dir", default="runs/isaac_go2")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--skills-per-episode", type=int, default=80)
parser.add_argument("--skill-horizon", type=int, default=24, help="Isaac env steps to hold each high-level skill.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--epsilon", type=float, default=0.12)
parser.add_argument("--ucb-c", type=float, default=0.7)
parser.add_argument("--curiosity-weight", type=float, default=0.2)
parser.add_argument("--uncertainty-weight", type=float, default=0.35)
parser.add_argument("--warmup-transitions", type=int, default=32)
parser.add_argument("--model-min-replay", type=int, default=64)
parser.add_argument("--sequence-length", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--train-every-skills", type=int, default=4)
parser.add_argument("--train-batches", type=int, default=1)
parser.add_argument("--checkpoint-every", type=int, default=25)
parser.add_argument("--video", action="store_true", default=False, help="Record an mp4 of the Isaac Lab rollout.")
parser.add_argument("--video-length", type=int, default=900, help="Maximum recorded video length in Isaac env steps.")
parser.add_argument("--video-dir", default=None, help="Directory for recorded mp4 files. Defaults under storage-dir.")
parser.add_argument(
    "--camera-mode",
    choices=["auto", "follow", "group"],
    default="auto",
    help="Camera behavior: follow agent 0, frame all agents, or choose group when num_envs > 1.",
)
parser.add_argument("--camera-distance", type=float, default=2.6, help="Base camera distance from the Go2 agents.")
parser.add_argument("--camera-height", type=float, default=1.45, help="Base camera height above the Go2 agents.")
parser.add_argument(
    "--cooperative-goal",
    action="store_true",
    default=False,
    help="Coordinate the Go2 agents toward a shared rally target while preserving formation slots.",
)
parser.add_argument("--goal-offset-x", type=float, default=0.65, help="Shared goal offset from the initial team centroid.")
parser.add_argument("--goal-offset-y", type=float, default=0.0, help="Shared goal offset from the initial team centroid.")
parser.add_argument("--goal-radius", type=float, default=0.35, help="Centroid radius for declaring the team goal reached.")
parser.add_argument("--slot-tolerance", type=float, default=0.55, help="Mean slot error tolerance for formation success.")
parser.add_argument(
    "--cooperation-weight",
    type=float,
    default=5.0,
    help="Weight of cooperative goal progress in high-level skill selection.",
)
parser.add_argument("--cooperative-centroid-weight", type=float, default=1.25)
parser.add_argument("--cooperative-slot-weight", type=float, default=1.0)
parser.add_argument("--low-level-reward-weight", type=float, default=0.1)
parser.add_argument("--model-dim", type=int, default=128)
parser.add_argument("--model-layers", type=int, default=3)
parser.add_argument("--model-heads", type=int, default=4)
parser.add_argument("--latent-dim", type=int, default=16)
parser.add_argument("--ssm-state-dim", type=int, default=16)
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--policy-checkpoint", default=None, help="RSL-RL checkpoint path for the Go2 low-level policy.")
parser.add_argument(
    "--pretrained-policy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fetch Isaac Lab's published RSL-RL checkpoint when no checkpoint path is supplied.",
)
parser.add_argument(
    "--allow-zero-policy",
    action="store_true",
    help="Permit a zero-action fallback if no RSL-RL policy is available. Useful only for plumbing checks.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils import parse_env_cfg

from mindcraft.robotics import (
    ROBOTICS_SKILLS,
    CooperativeGoalMetrics,
    RoboticsReplayBuffer,
    RoboticsSkillLibrary,
    RoboticsTransition,
    RoboticsWorldModelTrainer,
    choose_robotics_skill,
    cooperative_goal_metrics,
    cooperative_skill_score,
)


class GymVecAdapter:
    def __init__(self, env: gym.Env):
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.action_space = env.action_space

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.env.reset()

    def get_observations(self) -> Any:
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        return self.unwrapped._get_observations()

    def step(self, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        return obs, reward, (terminated | truncated).to(dtype=torch.long), extras

    def close(self) -> None:
        self.env.close()


class ZeroPolicy:
    def __init__(self, action_shape: tuple[int, ...], device: torch.device | str):
        self.action_shape = action_shape
        self.device = torch.device(device)

    def __call__(self, _obs: Any) -> torch.Tensor:
        return torch.zeros(self.action_shape, device=self.device)


class LegacyRslActorPolicy:
    """Inference-only loader for older RSL-RL checkpoints with model_state_dict actor.* keys."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location=self.device)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict) or "actor.0.weight" not in state:
            raise ValueError("checkpoint is not a legacy RSL-RL actor checkpoint")
        layer_indices = sorted(
            int(key.split(".")[1])
            for key in state
            if key.startswith("actor.") and key.endswith(".weight") and key.split(".")[1].isdigit()
        )
        modules: list[nn.Module] = []
        for index, layer_index in enumerate(layer_indices):
            weight = state[f"actor.{layer_index}.weight"]
            bias = state[f"actor.{layer_index}.bias"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            modules.append(layer)
            if index < len(layer_indices) - 1:
                modules.append(nn.ELU())
        self.net = nn.Sequential(*modules).to(self.device).eval()

    def __call__(self, obs: Any) -> torch.Tensor:
        with torch.inference_mode():
            return self.net(policy_tensor(obs).to(self.device).float())


def make_vec_env_and_policy(env: gym.Env) -> tuple[Any, Any, str]:
    try:
        import importlib.metadata as metadata

        from packaging import version
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    except Exception as exc:
        if args_cli.allow_zero_policy:
            vec_env = GymVecAdapter(env)
            return vec_env, ZeroPolicy(vec_env.action_space.shape, vec_env.device), f"zero_policy_missing_rsl_rl:{exc}"
        raise RuntimeError("RSL-RL policy support is unavailable; pass --allow-zero-policy for a plumbing check") from exc

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    try:
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    except Exception:
        installed_version = "unknown"

    if getattr(args_cli, "device", None) is not None and hasattr(agent_cfg, "device"):
        agent_cfg.device = args_cli.device
    vec_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    checkpoint = args_cli.policy_checkpoint
    if checkpoint is None and args_cli.pretrained_policy:
        task_name = args_cli.task.split(":")[-1].replace("-Play", "")
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", task_name)
    if checkpoint is None:
        if not args_cli.allow_zero_policy:
            raise RuntimeError(
                "No Go2 RSL-RL policy checkpoint was supplied or published for this task. "
                "Pass --policy-checkpoint /path/to/checkpoint.pt, or add --allow-zero-policy for a plumbing check."
            )
        return vec_env, ZeroPolicy(vec_env.action_space.shape, vec_env.device), "zero_policy_no_checkpoint"

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise RuntimeError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")
    try:
        runner.load(str(checkpoint))
    except Exception as exc:
        try:
            legacy_policy = LegacyRslActorPolicy(checkpoint, vec_env.unwrapped.device)
            return vec_env, legacy_policy, f"legacy_rsl_actor:{checkpoint}:original_load_error={exc}"
        except Exception as legacy_exc:
            print(f"legacy checkpoint load also failed: {legacy_exc}")
        if args_cli.allow_zero_policy:
            print(f"checkpoint load failed; using zero-action fallback because --allow-zero-policy is set: {exc}")
            return vec_env, ZeroPolicy(vec_env.action_space.shape, vec_env.device), f"zero_policy_checkpoint_load_failed:{exc}"
        raise
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    return vec_env, policy, f"rsl_rl:{checkpoint}:version={installed_version}"


def policy_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict) or hasattr(obs, "__getitem__"):
        try:
            return obs["policy"]
        except Exception:
            pass
    if isinstance(obs, torch.Tensor):
        return obs
    raise TypeError(f"Cannot extract policy observation from {type(obs)}")


def learning_observation(vec_env: Any, obs: Any) -> np.ndarray:
    env = vec_env.unwrapped
    policy_obs = policy_tensor(obs).float()
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    root_lin_vel = robot.data.root_lin_vel_b
    root_ang_vel = robot.data.root_ang_vel_b
    command = env.command_manager.get_command("base_velocity")
    root_features = torch.cat(
        [
            torch.tanh(root_pos[:, :2] / 10.0),
            torch.tanh((root_pos[:, 2:3] - 0.4) / 2.0),
            root_quat,
            torch.tanh(root_lin_vel / 3.0),
            torch.tanh(root_ang_vel / 3.0),
            command,
        ],
        dim=-1,
    )
    fused = torch.cat([policy_obs, root_features], dim=-1)
    return torch.nan_to_num(fused).detach().cpu().numpy().astype(np.float32)


def root_xy(vec_env: Any) -> np.ndarray:
    return vec_env.unwrapped.scene["robot"].data.root_pos_w[:, :2].detach().cpu().numpy()


def root_yaw(vec_env: Any) -> np.ndarray:
    quat = vec_env.unwrapped.scene["robot"].data.root_quat_w.detach().cpu().numpy()
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def update_camera(vec_env: Any, extra_xy: np.ndarray | None = None) -> None:
    robot = vec_env.unwrapped.scene["robot"]
    roots = robot.data.root_pos_w.detach().cpu().numpy()
    camera_mode = args_cli.camera_mode
    if camera_mode == "auto":
        camera_mode = "group" if roots.shape[0] > 1 else "follow"
    if camera_mode == "group":
        frame_xy = roots[:, :2]
        if extra_xy is not None and extra_xy.size:
            frame_xy = np.concatenate([frame_xy, np.asarray(extra_xy, dtype=np.float32).reshape(-1, 2)], axis=0)
        xy_min = frame_xy.min(axis=0)
        xy_max = frame_xy.max(axis=0)
        center_xy = 0.5 * (xy_min + xy_max)
        center_z = float(roots[:, 2].mean())
        span = float(max(xy_max[0] - xy_min[0], xy_max[1] - xy_min[1], 1.0))
        distance = args_cli.camera_distance + 0.85 * span
        height = args_cli.camera_height + 0.55 * span
        target = [float(center_xy[0]), float(center_xy[1]), center_z + 0.25]
        eye = [
            float(center_xy[0] - distance),
            float(center_xy[1] - 0.9 * distance),
            center_z + height,
        ]
        vec_env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        return
    root = roots[0]
    target = [float(root[0]), float(root[1]), float(root[2] + 0.25)]
    eye = [
        float(root[0] - args_cli.camera_distance),
        float(root[1] - 0.9 * args_cli.camera_distance),
        float(root[2] + args_cli.camera_height),
    ]
    vec_env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def goal_frame_points(goal_xy: np.ndarray | None, slot_offsets: np.ndarray | None) -> np.ndarray | None:
    if goal_xy is None:
        return None
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(1, 2)
    if slot_offsets is None:
        return goal
    return np.concatenate([goal, goal + np.asarray(slot_offsets, dtype=np.float32).reshape(-1, 2)], axis=0)


def make_goal_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CooperativeGoal",
        markers={
            "goal_active": sim_utils.CylinderCfg(
                radius=1.0,
                height=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.22)),
            ),
            "goal_achieved": sim_utils.CylinderCfg(
                radius=1.0,
                height=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.12)),
            ),
            "slot": sim_utils.SphereCfg(
                radius=0.09,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.45, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_goal_visualizer(
    visualizer: VisualizationMarkers | None,
    goal_xy: np.ndarray | None,
    slot_offsets: np.ndarray | None,
    achieved: bool,
) -> None:
    if visualizer is None or goal_xy is None:
        return
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    slot_targets = goal[None, :] + np.asarray(slot_offsets if slot_offsets is not None else [], dtype=np.float32).reshape(-1, 2)
    translations = [[float(goal[0]), float(goal[1]), 0.025]]
    translations.extend([[float(target[0]), float(target[1]), 0.16] for target in slot_targets])
    scales = [[float(args_cli.goal_radius), float(args_cli.goal_radius), 1.0]]
    scales.extend([[1.0, 1.0, 1.0] for _ in slot_targets])
    marker_indices = [1 if achieved else 0]
    marker_indices.extend([2] * len(slot_targets))
    visualizer.visualize(
        translations=np.asarray(translations, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        marker_indices=np.asarray(marker_indices, dtype=np.int32),
    )


def base_skill_scores(
    observation: np.ndarray,
    library: RoboticsSkillLibrary,
    trainer: RoboticsWorldModelTrainer,
    replay_size: int,
) -> tuple[dict[str, float], str, dict[str, Any]]:
    library_scores = library.score_candidates(ucb_c=args_cli.ucb_c, curiosity_weight=args_cli.curiosity_weight)
    if replay_size < args_cli.model_min_replay:
        return library_scores, "skill_library", {
            skill: {"score": score, "library_score": score} for skill, score in library_scores.items()
        }
    diagnostics: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for skill in ROBOTICS_SKILLS:
        prediction = trainer.predict_skill(observation, skill)
        score = (
            library_scores.get(skill, 0.0)
            + prediction["reward"]
            + trainer.gamma * prediction["value"]
            + 0.1 * prediction["prior"]
            - args_cli.uncertainty_weight * prediction["model_uncertainty"]
        )
        scores[skill] = float(score)
        diagnostics[skill] = {
            "score": float(score),
            "library_score": library_scores.get(skill, 0.0),
            "reward": prediction["reward"],
            "value": prediction["value"],
            "uncertainty": prediction["model_uncertainty"],
            "prior": prediction["prior"],
        }
    return scores, "robotics_world_model", diagnostics


def choose_cooperative_skills(
    current_obs: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    goal_xy: np.ndarray,
    slot_offsets: np.ndarray,
    library: RoboticsSkillLibrary,
    trainer: RoboticsWorldModelTrainer,
    replay_size: int,
    rng: random.Random,
    *,
    skill_horizon_s: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    selected: list[str] = []
    planner_names: list[str] = []
    env0_diagnostics: dict[str, Any] = {}
    for env_id in range(current_obs.shape[0]):
        if rng.random() < args_cli.epsilon:
            selected.append(rng.choice(tuple(ROBOTICS_SKILLS)))
            planner_names.append("cooperative_epsilon")
            continue
        scores, base_planner, diagnostics = base_skill_scores(current_obs[env_id], library, trainer, replay_size)
        combined: dict[str, dict[str, float]] = {}
        best_skill = next(iter(ROBOTICS_SKILLS))
        best_score = float("-inf")
        for skill, base_score in scores.items():
            coop = cooperative_skill_score(
                env_id,
                skill,
                positions,
                yaws,
                goal_xy,
                slot_offsets,
                skill_horizon_s=skill_horizon_s,
                goal_radius=args_cli.goal_radius,
                slot_tolerance=args_cli.slot_tolerance,
                centroid_weight=args_cli.cooperative_centroid_weight,
                slot_weight=args_cli.cooperative_slot_weight,
            )
            total_score = float(base_score) + args_cli.cooperation_weight * coop["score"]
            combined[skill] = {
                "score": total_score,
                "base_score": float(base_score),
                "cooperative_score": coop["score"],
                "centroid_progress": coop["centroid_progress"],
                "formation_progress": coop["formation_progress"],
                "slot_progress": coop["slot_progress"],
                "predicted_centroid_distance": coop["predicted_centroid_distance"],
                "predicted_mean_slot_error": coop["predicted_mean_slot_error"],
            }
            if total_score > best_score:
                best_score = total_score
                best_skill = skill
        selected.append(best_skill)
        planner_names.append(f"cooperative_{base_planner}")
        if env_id == 0:
            env0_diagnostics = combined or diagnostics
    return selected, planner_names, env0_diagnostics


def cooperative_transition_reward(
    low_level_reward: float,
    before: CooperativeGoalMetrics | None,
    after: CooperativeGoalMetrics | None,
    env_id: int,
    done: bool,
) -> tuple[float, dict[str, float | bool]]:
    if before is None or after is None:
        return low_level_reward, {}
    before_slot = before.slot_errors[env_id] if env_id < len(before.slot_errors) else before.mean_slot_error
    after_slot = after.slot_errors[env_id] if env_id < len(after.slot_errors) else after.mean_slot_error
    centroid_progress = before.centroid_distance - after.centroid_distance
    formation_progress = before.mean_slot_error - after.mean_slot_error
    slot_progress = before_slot - after_slot
    achievement_bonus = 2.0 if after.achieved else 0.0
    reward = (
        args_cli.low_level_reward_weight * low_level_reward
        + 2.0 * centroid_progress
        + 0.75 * formation_progress
        + 1.25 * slot_progress
        + achievement_bonus
        - (1.0 if done else 0.0)
    )
    return float(reward), {
        "low_level_reward": float(low_level_reward),
        "centroid_progress": float(centroid_progress),
        "formation_progress": float(formation_progress),
        "slot_progress": float(slot_progress),
        "slot_error": float(after_slot),
        "cooperative_goal_achieved": after.achieved,
    }


def set_velocity_commands(vec_env: Any, commands: torch.Tensor) -> None:
    term = vec_env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = commands
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e6


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def run() -> None:
    rng = random.Random(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    storage_dir = Path(args_cli.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    video_dir = Path(args_cli.video_dir) if args_cli.video_dir else storage_dir / "videos"
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "go2-continual-learning",
            "disable_logger": True,
        }
        print(f"recording Isaac Lab video to {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    vec_env, low_level_policy, policy_source = make_vec_env_and_policy(env)
    env_step_dt = float(getattr(vec_env.unwrapped, "step_dt", 0.02))
    skill_horizon_s = args_cli.skill_horizon * env_step_dt
    goal_visualizer = make_goal_visualizer() if args_cli.cooperative_goal else None

    obs = vec_env.get_observations()
    update_camera(vec_env)
    current_obs = learning_observation(vec_env, obs)
    trainer = RoboticsWorldModelTrainer(
        storage_dir,
        obs_dim=current_obs.shape[1],
        d_model=args_cli.model_dim,
        layers=args_cli.model_layers,
        heads=args_cli.model_heads,
        latent_dim=args_cli.latent_dim,
        ssm_state_dim=args_cli.ssm_state_dim,
        lora_rank=args_cli.lora_rank,
        lr=args_cli.lr,
        device=args_cli.device,
    )
    replay = RoboticsReplayBuffer(storage_dir / "go2_experience.jsonl")
    library = RoboticsSkillLibrary(storage_dir / "go2_skill_library.json")
    metrics_path = storage_dir / "go2_training_metrics.jsonl"
    orchestration_path = storage_dir / "go2_multiagent_orchestration.jsonl"
    goal_summary_path = storage_dir / "go2_cooperative_goal_summary.json"
    latest_goal_summary: dict[str, Any] | None = None

    print(f"Go2 continual demo task={args_cli.task} envs={vec_env.num_envs} policy={policy_source}")
    print(f"world_model_device={trainer.device} obs_dim={trainer.obs_dim} storage={storage_dir}")

    global_skill_step = 0
    try:
        for episode in range(args_cli.episodes):
            obs, _extras = vec_env.reset()
            current_obs = learning_observation(vec_env, obs)
            goal_xy: np.ndarray | None = None
            slot_offsets: np.ndarray | None = None
            goal_metrics: CooperativeGoalMetrics | None = None
            if args_cli.cooperative_goal:
                initial_xy = root_xy(vec_env)
                initial_centroid = initial_xy.mean(axis=0)
                goal_xy = initial_centroid + np.asarray([args_cli.goal_offset_x, args_cli.goal_offset_y], dtype=np.float32)
                slot_offsets = initial_xy - initial_centroid[None, :]
                goal_metrics = cooperative_goal_metrics(
                    initial_xy,
                    goal_xy,
                    slot_offsets,
                    goal_radius=args_cli.goal_radius,
                    slot_tolerance=args_cli.slot_tolerance,
                )
                update_goal_visualizer(goal_visualizer, goal_xy, slot_offsets, goal_metrics.achieved)
                latest_goal_summary = {
                    "episode": episode,
                    "initial": goal_metrics.to_jsonable(),
                    "latest": goal_metrics.to_jsonable(),
                    "achieved_step": None,
                    "skill_horizon_s": skill_horizon_s,
                }
                print(
                    "cooperative_goal "
                    f"target=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) "
                    f"initial_dist={goal_metrics.centroid_distance:.3f} "
                    f"slots={vec_env.num_envs}"
                )
            update_camera(vec_env, goal_frame_points(goal_xy, slot_offsets))
            for skill_step in range(args_cli.skills_per_episode):
                if not simulation_app.is_running():
                    break
                before_xy = root_xy(vec_env)
                selected: list[str] = []
                planner_names: list[str] = []
                diagnostics: dict[str, Any] = {}
                before_goal_metrics: CooperativeGoalMetrics | None = None
                if args_cli.cooperative_goal and goal_xy is not None and slot_offsets is not None:
                    before_goal_metrics = cooperative_goal_metrics(
                        before_xy,
                        goal_xy,
                        slot_offsets,
                        goal_radius=args_cli.goal_radius,
                        slot_tolerance=args_cli.slot_tolerance,
                    )
                    selected, planner_names, diagnostics = choose_cooperative_skills(
                        current_obs,
                        before_xy,
                        root_yaw(vec_env),
                        goal_xy,
                        slot_offsets,
                        library,
                        trainer,
                        len(replay),
                        rng,
                        skill_horizon_s=skill_horizon_s,
                    )
                else:
                    for env_id in range(vec_env.num_envs):
                        skill, planner_name, diag = choose_robotics_skill(
                            current_obs[env_id],
                            library,
                            rng,
                            world_model=trainer,
                            replay_size=len(replay),
                            epsilon=args_cli.epsilon,
                            model_min_replay=args_cli.model_min_replay,
                            ucb_c=args_cli.ucb_c,
                            curiosity_weight=args_cli.curiosity_weight,
                            uncertainty_weight=args_cli.uncertainty_weight,
                        )
                        selected.append(skill)
                        planner_names.append(planner_name)
                        if diag and env_id == 0:
                            diagnostics = diag

                commands = torch.tensor(
                    [ROBOTICS_SKILLS[name].command for name in selected],
                    dtype=torch.float32,
                    device=vec_env.unwrapped.device,
                )
                set_velocity_commands(vec_env, commands)
                accumulated_reward = torch.zeros(vec_env.num_envs, device=vec_env.unwrapped.device)
                done = torch.zeros(vec_env.num_envs, dtype=torch.bool, device=vec_env.unwrapped.device)
                for _ in range(args_cli.skill_horizon):
                    with torch.inference_mode():
                        set_velocity_commands(vec_env, commands)
                        if goal_xy is not None and slot_offsets is not None:
                            update_goal_visualizer(goal_visualizer, goal_xy, slot_offsets, bool(goal_metrics and goal_metrics.achieved))
                        update_camera(vec_env, goal_frame_points(goal_xy, slot_offsets))
                        policy_obs = vec_env.get_observations()
                        actions = low_level_policy(policy_obs)
                        obs, reward, dones, _extras = vec_env.step(actions)
                    accumulated_reward += reward
                    done |= dones.bool()

                next_obs = learning_observation(vec_env, obs)
                after_xy = root_xy(vec_env)
                avg_reward = (accumulated_reward / max(1, args_cli.skill_horizon)).detach().cpu().numpy()
                done_np = done.detach().cpu().numpy().astype(bool)
                delta_xy = after_xy - before_xy
                after_goal_metrics: CooperativeGoalMetrics | None = None
                if args_cli.cooperative_goal and goal_xy is not None and slot_offsets is not None:
                    after_goal_metrics = cooperative_goal_metrics(
                        after_xy,
                        goal_xy,
                        slot_offsets,
                        goal_radius=args_cli.goal_radius,
                        slot_tolerance=args_cli.slot_tolerance,
                    )
                    goal_metrics = after_goal_metrics
                    update_goal_visualizer(goal_visualizer, goal_xy, slot_offsets, after_goal_metrics.achieved)
                    if latest_goal_summary is not None:
                        latest_goal_summary["latest"] = after_goal_metrics.to_jsonable()
                        if after_goal_metrics.achieved and latest_goal_summary.get("achieved_step") is None:
                            latest_goal_summary["achieved_step"] = skill_step
                agent_records: list[dict[str, Any]] = []
                for env_id, skill in enumerate(selected):
                    command = ROBOTICS_SKILLS[skill].command
                    command_xy = np.asarray(command[:2], dtype=np.float32)
                    command_norm = float(np.linalg.norm(command_xy))
                    progress = float(np.dot(delta_xy[env_id], command_xy / command_norm)) if command_norm > 1.0e-6 else 0.0
                    transition_reward, cooperative_info = cooperative_transition_reward(
                        float(avg_reward[env_id]),
                        before_goal_metrics,
                        after_goal_metrics,
                        env_id,
                        bool(done_np[env_id]),
                    )
                    if after_goal_metrics is not None:
                        success = bool(not done_np[env_id] and (after_goal_metrics.achieved or transition_reward > 0.0))
                    else:
                        success = bool(not done_np[env_id] and (avg_reward[env_id] > 0.0 or progress > 0.02))
                    transition = RoboticsTransition(
                        episode=episode,
                        step=skill_step,
                        env_id=env_id,
                        skill=skill,
                        command=command,
                        observation=current_obs[env_id].tolist(),
                        reward=transition_reward,
                        next_observation=next_obs[env_id].tolist(),
                        done=bool(done_np[env_id]),
                        info={
                            "planner": planner_names[env_id],
                            "low_level_reward": float(avg_reward[env_id]),
                            "progress_m": progress,
                            "delta_xy": delta_xy[env_id].tolist(),
                            "cooperative": cooperative_info,
                        },
                    )
                    prediction_error = None
                    if len(replay) >= args_cli.warmup_transitions:
                        prediction_error = trainer.prediction_error(transition)
                    replay.append(transition)
                    library.update(
                        skill=skill,
                        reward=transition.reward,
                        success=success,
                        episode=episode,
                        step=skill_step,
                        prediction_error=prediction_error,
                        note=f"reward={transition.reward:.3f} progress={progress:.3f}",
                    )
                    agent_records.append(
                        {
                            "env_id": env_id,
                            "skill": skill,
                            "planner": planner_names[env_id],
                            "command": list(command),
                            "reward": transition.reward,
                            "low_level_reward": float(avg_reward[env_id]),
                            "progress_m": progress,
                            "success": success,
                            "done": transition.done,
                            "prediction_error": prediction_error,
                            "cooperative": cooperative_info,
                        }
                    )

                current_obs = next_obs
                global_skill_step += 1
                metrics = None
                if (
                    len(replay) >= args_cli.warmup_transitions
                    and replay.can_sample_sequence(args_cli.sequence_length)
                    and global_skill_step % args_cli.train_every_skills == 0
                ):
                    for _ in range(args_cli.train_batches):
                        seqs = replay.sample_sequences(args_cli.batch_size, args_cli.sequence_length, rng)
                        val = replay.sample_sequences(
                            max(1, args_cli.batch_size // 4),
                            args_cli.sequence_length,
                            rng,
                            holdout=True,
                        )
                        metrics = trainer.train_batches(seqs, validation_sequences=val)
                    if metrics is not None:
                        append_jsonl(
                            metrics_path,
                            {
                                "phase": "online_go2",
                                "episode": episode,
                                "skill_step": skill_step,
                                "global_skill_step": global_skill_step,
                                "replay_size": len(replay),
                                "metrics": asdict(metrics),
                            },
                        )

                skill_counts: dict[str, int] = {}
                planner_counts: dict[str, int] = {}
                for skill in selected:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
                for planner_name in planner_names:
                    planner_counts[planner_name] = planner_counts.get(planner_name, 0) + 1
                append_jsonl(
                    orchestration_path,
                    {
                        "phase": (
                            "cooperative_goal_go2"
                            if args_cli.cooperative_goal
                            else "multi_agent_go2"
                            if vec_env.num_envs > 1
                            else "single_agent_go2"
                        ),
                        "episode": episode,
                        "skill_step": skill_step,
                        "global_skill_step": global_skill_step,
                        "num_agents": vec_env.num_envs,
                        "shared_replay_size": len(replay),
                        "world_model_train_step": trainer.train_step,
                        "mean_reward": float(np.mean([record["reward"] for record in agent_records])),
                        "mean_low_level_reward": float(np.mean(avg_reward)),
                        "skill_counts": skill_counts,
                        "planner_counts": planner_counts,
                        "cooperative_goal_before": before_goal_metrics.to_jsonable()
                        if before_goal_metrics is not None
                        else None,
                        "cooperative_goal_after": after_goal_metrics.to_jsonable()
                        if after_goal_metrics is not None
                        else None,
                        "agents": agent_records,
                    },
                )

                if global_skill_step % args_cli.checkpoint_every == 0:
                    trainer.save()
                    library.save()

                first_skill = selected[0]
                first_reward = float(agent_records[0]["reward"]) if agent_records else float(avg_reward[0])
                agent_trace = " ".join(
                    f"env{env_id}:{selected[env_id]}/{planner_names[env_id]}/{agent_records[env_id]['reward']:.3f}"
                    for env_id in range(min(vec_env.num_envs, 6))
                )
                suffix = " ..." if vec_env.num_envs > 6 else ""
                if vec_env.num_envs == 1:
                    agent_trace = f"skill={first_skill} planner={planner_names[0]} reward={first_reward:.3f}"
                goal_text = ""
                if after_goal_metrics is not None:
                    goal_text = (
                        f" goal_dist={after_goal_metrics.centroid_distance:.3f}"
                        f" slot_err={after_goal_metrics.mean_slot_error:.3f}"
                        f" achieved={after_goal_metrics.achieved}"
                    )
                print(
                    f"[{episode}:{skill_step}] {agent_trace}{suffix} "
                    f"shared_replay={len(replay)} train_step={trainer.train_step}{goal_text}"
                )
                if diagnostics and global_skill_step % 10 == 0:
                    best = sorted(diagnostics.items(), key=lambda item: item[1]["score"], reverse=True)[:3]
                    print("  model_top=" + json.dumps(best, sort_keys=True))
    finally:
        if latest_goal_summary is not None:
            goal_summary_path.write_text(json.dumps(latest_goal_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        library.save()
        trainer.save()
        vec_env.close()
        print(f"saved Go2 replay/library/world model under {storage_dir}")


if __name__ == "__main__":
    run()
    simulation_app.close()
