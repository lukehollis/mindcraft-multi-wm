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
parser.add_argument("--camera-distance", type=float, default=2.6, help="Chase-camera distance from the first Go2 robot.")
parser.add_argument("--camera-height", type=float, default=1.45, help="Chase-camera height above the first Go2 robot.")
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from mindcraft.robotics import (
    ROBOTICS_SKILLS,
    RoboticsReplayBuffer,
    RoboticsSkillLibrary,
    RoboticsTransition,
    RoboticsWorldModelTrainer,
    choose_robotics_skill,
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


def update_camera(vec_env: Any) -> None:
    robot = vec_env.unwrapped.scene["robot"]
    root = robot.data.root_pos_w[0].detach().cpu().numpy()
    target = [float(root[0]), float(root[1]), float(root[2] + 0.25)]
    eye = [
        float(root[0] - args_cli.camera_distance),
        float(root[1] - 0.9 * args_cli.camera_distance),
        float(root[2] + args_cli.camera_height),
    ]
    vec_env.unwrapped.sim.set_camera_view(eye=eye, target=target)


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

    print(f"Go2 continual demo task={args_cli.task} envs={vec_env.num_envs} policy={policy_source}")
    print(f"world_model_device={trainer.device} obs_dim={trainer.obs_dim} storage={storage_dir}")

    global_skill_step = 0
    try:
        for episode in range(args_cli.episodes):
            obs, _extras = vec_env.reset()
            update_camera(vec_env)
            current_obs = learning_observation(vec_env, obs)
            for skill_step in range(args_cli.skills_per_episode):
                if not simulation_app.is_running():
                    break
                before_xy = root_xy(vec_env)
                selected: list[str] = []
                planner_names: list[str] = []
                diagnostics: dict[str, Any] = {}
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
                        update_camera(vec_env)
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
                for env_id, skill in enumerate(selected):
                    command = ROBOTICS_SKILLS[skill].command
                    command_xy = np.asarray(command[:2], dtype=np.float32)
                    command_norm = float(np.linalg.norm(command_xy))
                    progress = float(np.dot(delta_xy[env_id], command_xy / command_norm)) if command_norm > 1.0e-6 else 0.0
                    success = bool(not done_np[env_id] and (avg_reward[env_id] > 0.0 or progress > 0.02))
                    transition = RoboticsTransition(
                        episode=episode,
                        step=skill_step,
                        env_id=env_id,
                        skill=skill,
                        command=command,
                        observation=current_obs[env_id].tolist(),
                        reward=float(avg_reward[env_id]),
                        next_observation=next_obs[env_id].tolist(),
                        done=bool(done_np[env_id]),
                        info={
                            "planner": planner_names[env_id],
                            "progress_m": progress,
                            "delta_xy": delta_xy[env_id].tolist(),
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

                if global_skill_step % args_cli.checkpoint_every == 0:
                    trainer.save()
                    library.save()

                first_skill = selected[0]
                first_reward = float(avg_reward[0])
                print(
                    f"[{episode}:{skill_step}] skill={first_skill} planner={planner_names[0]} "
                    f"reward={first_reward:.3f} replay={len(replay)} train_step={trainer.train_step}"
                )
                if diagnostics and global_skill_step % 10 == 0:
                    best = sorted(diagnostics.items(), key=lambda item: item[1]["score"], reverse=True)[:3]
                    print("  model_top=" + json.dumps(best, sort_keys=True))
    finally:
        library.save()
        trainer.save()
        vec_env.close()
        print(f"saved Go2 replay/library/world model under {storage_dir}")


if __name__ == "__main__":
    run()
    simulation_app.close()
