from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mindcraft.skills import SKILLS, WOOD_LOG_ITEMS, WOOD_PLANK_ITEMS
from mindcraft.schemas import Observation, Transition
from mindcraft.progression import PROGRESSION_ITEMS, skill_affordance_mask, snapshot as progression_snapshot


INVENTORY_KEYS = (
    *WOOD_LOG_ITEMS,
    *WOOD_PLANK_ITEMS,
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "stone_pickaxe",
    "coal",
    "iron_ore",
    "furnace",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
)

BLOCK_KEYS = (
    "oak_log",
    "birch_log",
    "spruce_log",
    "stone",
    "coal_ore",
    "deepslate_coal_ore",
    "iron_ore",
    "deepslate_iron_ore",
    "diamond_ore",
    "deepslate_diamond_ore",
    "water",
    "lava",
)

SKILL_NAMES = tuple(SKILLS)
UNLOCK_KEYS = tuple(PROGRESSION_ITEMS)
ACTION_ALIASES = {
    "scout_area": "explore_area",
}
OBS_DIM = 5 + len(INVENTORY_KEYS) + len(BLOCK_KEYS)
ACTION_DIM = len(SKILL_NAMES)
UNLOCK_DIM = len(UNLOCK_KEYS)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class LoRALinear(nn.Module):
    """Linear layer with an optional low-rank adapter for cheap online adaptation."""

    def __init__(self, in_features: int, out_features: int, rank: int = 0, bias: bool = True, alpha: float = 1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        if self.rank > 0:
            self.lora_a = nn.Linear(in_features, self.rank, bias=False)
            self.lora_b = nn.Linear(self.rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_a is None or self.lora_b is None:
            return y
        return y + (self.alpha / self.rank) * self.lora_b(self.lora_a(x))

    def freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False


class ScalarEnsembleHead(nn.Module):
    def __init__(self, in_features: int, members: int = 3, lora_rank: int = 0):
        super().__init__()
        self.members = nn.ModuleList(
            [LoRALinear(in_features, 1, rank=lora_rank) for _ in range(max(1, members))]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = torch.stack([member(x).squeeze(-1) for member in self.members], dim=-1)
        mean = predictions.mean(dim=-1)
        if predictions.shape[-1] == 1:
            uncertainty = torch.zeros_like(mean)
        else:
            uncertainty = predictions.std(dim=-1, unbiased=False)
        return mean, uncertainty, predictions


class SelectiveSSMBlock(nn.Module):
    """
    A pure PyTorch Mamba-2-inspired selective SSM block.

    The native Mamba kernels are faster, but this recurrent scan keeps the repo runnable on
    CPU, Docker, and DGX without compiling custom CUDA extensions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 3,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = RMSNorm(d_model)
        self.in_proj = LoRALinear(d_model, self.d_inner * 2, rank=lora_rank, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,
        )
        self.param_proj = LoRALinear(self.d_inner, self.d_inner + 2 * d_state, rank=lora_rank, bias=True)
        self.out_proj = LoRALinear(self.d_inner, d_model, rank=lora_rank, bias=False)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        u = F.silu(u)
        params = self.param_proj(u)
        d_state = self.a_log.shape[-1]
        dt, b, c = torch.split(params, [self.d_inner, d_state, d_state], dim=-1)
        dt = F.softplus(dt).clamp(max=1.0)
        a = -torch.exp(self.a_log).to(u.dtype)
        y = selective_scan(u, dt, a, b, c, self.d_skip)
        y = y * torch.sigmoid(gate)
        return residual + self.out_proj(y)


def selective_scan(
    u: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_skip: torch.Tensor,
) -> torch.Tensor:
    batch, seq, d_inner = u.shape
    state = torch.zeros(batch, d_inner, a.shape[-1], device=u.device, dtype=u.dtype)
    outputs: list[torch.Tensor] = []
    for t in range(seq):
        dt_t = dt[:, t].unsqueeze(-1)
        u_t = u[:, t].unsqueeze(-1)
        decay = torch.exp(dt_t * a.unsqueeze(0))
        state = decay * state + dt_t * b[:, t].unsqueeze(1) * u_t
        y_t = torch.sum(state * c[:, t].unsqueeze(1), dim=-1) + d_skip.unsqueeze(0) * u[:, t]
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)


def lerp_matching_prefix(target: torch.Tensor, source: torch.Tensor, tau: float) -> None:
    if target.shape == source.shape:
        target.lerp_(source, tau)
        return
    slices = tuple(slice(0, min(t_dim, s_dim)) for t_dim, s_dim in zip(target.shape, source.shape))
    target[slices].lerp_(source[slices], tau)


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim: int = 6, num_bins: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("basis", num_bins ** torch.arange(latent_dim, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bounded = 0.5 * (torch.tanh(z) + 1.0) * (self.num_bins - 1)
        rounded = torch.round(bounded)
        straight_through = bounded + (rounded - bounded).detach()
        return 2.0 * straight_through / (self.num_bins - 1) - 1.0

    def indices(self, z: torch.Tensor) -> torch.Tensor:
        bounded = 0.5 * (z + 1.0) * (self.num_bins - 1)
        digits = torch.round(bounded).clamp(0, self.num_bins - 1)
        return torch.sum(digits.long() * self.basis.to(z.device), dim=-1)


class ActionConditionedWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        latent_dim: int = 16,
        ssm_state_dim: int = 16,
        ensemble_size: int = 3,
        lora_rank: int = 0,
        freeze_base_for_lora: bool = False,
    ):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            LoRALinear(obs_dim, d_model, rank=lora_rank),
            nn.GELU(),
            LoRALinear(d_model, d_model, rank=lora_rank),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )
        self.action_encoder = LoRALinear(action_dim, d_model, rank=lora_rank)
        self.blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=d_model,
                    d_state=ssm_state_dim,
                    expand=max(1, heads // 2),
                    lora_rank=lora_rank,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.latent_proj = LoRALinear(d_model, latent_dim, rank=lora_rank)
        self.fsq = FiniteScalarQuantizer(latent_dim=latent_dim, num_bins=8)
        self.dec_proj = LoRALinear(latent_dim, d_model, rank=lora_rank)
        self.jepa_predictor = nn.Sequential(
            LoRALinear(d_model, d_model, rank=lora_rank),
            nn.GELU(),
            LoRALinear(d_model, latent_dim, rank=lora_rank),
        )
        self.next_obs = LoRALinear(d_model, obs_dim, rank=lora_rank)
        self.reward = ScalarEnsembleHead(d_model, members=ensemble_size, lora_rank=lora_rank)
        self.done = LoRALinear(d_model, 1, rank=lora_rank)
        self.value = ScalarEnsembleHead(d_model, members=ensemble_size, lora_rank=lora_rank)
        self.policy = LoRALinear(d_model, action_dim, rank=lora_rank)
        self.unlock = LoRALinear(d_model, UNLOCK_DIM, rank=lora_rank)
        self.affordance = LoRALinear(d_model, action_dim, rank=lora_rank)
        self._reset_target_encoder()
        if freeze_base_for_lora and lora_rank > 0:
            self.freeze_base_for_lora()

    def _reset_target_encoder(self) -> None:
        with torch.no_grad():
            source = list(self.obs_encoder.modules())
            target = list(self.target_encoder.modules())
            linear_sources = [m for m in source if isinstance(m, LoRALinear)]
            linear_targets = [m for m in target if isinstance(m, nn.Linear)]
            for src, dst in zip(linear_sources[:2], linear_targets):
                lerp_matching_prefix(dst.weight, src.base.weight, tau=1.0)
                if dst.bias is not None and src.base.bias is not None:
                    lerp_matching_prefix(dst.bias, src.base.bias, tau=1.0)

    def freeze_base_for_lora(self) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.freeze_base()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, tau: float = 0.01) -> None:
        source_params = [
            param
            for module in self.obs_encoder.modules()
            if isinstance(module, LoRALinear)
            for param in module.base.parameters()
        ]
        target_params = list(self.target_encoder.parameters())
        for target, source in zip(target_params, source_params):
            lerp_matching_prefix(target, source.detach(), tau)

    def encode_state(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.obs_encoder(obs) + self.action_encoder(action)
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)

    def project_latent(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        continuous = self.latent_proj(h)
        quantized = self.fsq(continuous)
        decoded = h + self.dec_proj(quantized)
        return quantized, decoded

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encode_state(obs, action)
        z, decoded = self.project_latent(h)
        reward, reward_uncertainty, reward_members = self.reward(decoded)
        value, value_uncertainty, value_members = self.value(decoded)
        outputs = {
            "next_obs": self.next_obs(decoded),
            "reward": reward,
            "reward_uncertainty": reward_uncertainty,
            "reward_members": reward_members,
            "done": self.done(decoded).squeeze(-1),
            "value": value,
            "value_uncertainty": value_uncertainty,
            "value_members": value_members,
            "policy": self.policy(decoded),
            "unlock": self.unlock(decoded),
            "affordance": self.affordance(decoded),
            "pred_latent": self.jepa_predictor(decoded),
            "codes": self.fsq.indices(z),
        }
        if next_obs is not None:
            outputs["target_latent"] = self.target_encoder(next_obs).detach()
        return outputs


@dataclass(slots=True)
class WorldModelMetrics:
    train_step: int
    loss: float
    obs_loss: float
    jepa_loss: float
    reward_loss: float
    value_loss: float
    policy_loss: float
    unlock_loss: float
    affordance_loss: float
    done_loss: float
    code_usage: float
    val_loss: float | None = None
    val_obs_loss: float | None = None
    val_reward_loss: float | None = None
    val_value_loss: float | None = None
    val_policy_loss: float | None = None
    val_unlock_loss: float | None = None
    val_affordance_loss: float | None = None
    val_done_loss: float | None = None


class WorldModelTrainer:
    def __init__(
        self,
        storage_dir: Path,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        latent_dim: int = 16,
        ssm_state_dim: int = 16,
        ensemble_size: int = 3,
        lora_rank: int = 0,
        freeze_base_for_lora: bool = False,
        lr: float = 3.0e-4,
        gamma: float = 0.97,
        device: str | None = None,
        checkpoint_name: str = "world_model.pt",
    ):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.model_config = {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "latent_dim": latent_dim,
            "ssm_state_dim": ssm_state_dim,
            "ensemble_size": ensemble_size,
            "lora_rank": lora_rank,
            "freeze_base_for_lora": freeze_base_for_lora,
            "unlock_dim": UNLOCK_DIM,
        }
        self.model = ActionConditionedWorldModel(
            OBS_DIM,
            ACTION_DIM,
            d_model,
            layers,
            heads,
            latent_dim=latent_dim,
            ssm_state_dim=ssm_state_dim,
            ensemble_size=ensemble_size,
            lora_rank=lora_rank,
            freeze_base_for_lora=freeze_base_for_lora,
        ).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.last_loss = 1.0
        self.train_step = 0
        self.checkpoint_path = self.storage_dir / checkpoint_name
        self.checkpoint_meta_path = self.storage_dir / f"{self.checkpoint_path.stem}_checkpoint.json"
        self._loaded_checkpoint_signatures: dict[str, tuple[int, int]] = {}
        if self.checkpoint_path.exists():
            try:
                self.load()
            except RuntimeError as exc:
                reason = str(exc).splitlines()[0]
                if reason.startswith("Error(s) in loading state_dict"):
                    reason = "checkpoint architecture does not match current model"
                print(f"ignoring incompatible world model checkpoint {self.checkpoint_path}: {reason}")

    def train_batches(
        self,
        sequences: list[list[Transition]],
        validation_sequences: list[list[Transition]] | None = None,
    ) -> WorldModelMetrics | None:
        if not sequences:
            return None
        self.model.train()
        obs, action, next_obs, reward, done, unlock, affordance = batch_to_tensors(sequences, self.device)
        pred = self.model(obs, action, next_obs=next_obs)
        losses = self._loss_components(pred, action, next_obs, reward, done, unlock, affordance)
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
        return WorldModelMetrics(
            train_step=self.train_step,
            loss=self.last_loss,
            obs_loss=float(losses["obs_loss"].detach().cpu()),
            jepa_loss=float(losses["jepa_loss"].detach().cpu()),
            reward_loss=float(losses["reward_loss"].detach().cpu()),
            value_loss=float(losses["value_loss"].detach().cpu()),
            policy_loss=float(losses["policy_loss"].detach().cpu()),
            unlock_loss=float(losses["unlock_loss"].detach().cpu()),
            affordance_loss=float(losses["affordance_loss"].detach().cpu()),
            done_loss=float(losses["done_loss"].detach().cpu()),
            code_usage=float(code_usage),
            **validation,
        )

    @torch.no_grad()
    def evaluate_batches(self, sequences: list[list[Transition]]) -> dict[str, float]:
        if not sequences:
            return {}
        self.model.eval()
        obs, action, next_obs, reward, done, unlock, affordance = batch_to_tensors(sequences, self.device)
        pred = self.model(obs, action, next_obs=next_obs)
        losses = self._loss_components(pred, action, next_obs, reward, done, unlock, affordance)
        return {
            "val_loss": float(losses["loss"].detach().cpu()),
            "val_obs_loss": float(losses["obs_loss"].detach().cpu()),
            "val_reward_loss": float(losses["reward_loss"].detach().cpu()),
            "val_value_loss": float(losses["value_loss"].detach().cpu()),
            "val_policy_loss": float(losses["policy_loss"].detach().cpu()),
            "val_unlock_loss": float(losses["unlock_loss"].detach().cpu()),
            "val_affordance_loss": float(losses["affordance_loss"].detach().cpu()),
            "val_done_loss": float(losses["done_loss"].detach().cpu()),
        }

    def _loss_components(
        self,
        pred: dict[str, torch.Tensor],
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        unlock: torch.Tensor,
        affordance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        returns = discounted_returns(reward, done, self.gamma)
        policy_target = action.argmax(dim=-1)
        obs_loss = F.smooth_l1_loss(pred["next_obs"], next_obs)
        jepa_loss = latent_prediction_loss(pred["pred_latent"], pred["target_latent"])
        reward_target = reward.unsqueeze(-1).expand_as(pred["reward_members"])
        reward_loss = F.mse_loss(pred["reward_members"], reward_target)
        value_target = returns.unsqueeze(-1).expand_as(pred["value_members"])
        value_loss = F.smooth_l1_loss(pred["value_members"], value_target)
        policy_loss = F.cross_entropy(pred["policy"].reshape(-1, ACTION_DIM), policy_target.reshape(-1))
        unlock_loss = F.binary_cross_entropy_with_logits(pred["unlock"], unlock)
        affordance_loss = F.binary_cross_entropy_with_logits(pred["affordance"], affordance)
        done_loss = F.binary_cross_entropy_with_logits(pred["done"], done)
        loss = (
            obs_loss
            + jepa_loss
            + reward_loss
            + 0.5 * value_loss
            + 0.2 * policy_loss
            + 0.35 * unlock_loss
            + 0.15 * affordance_loss
            + 0.2 * done_loss
        )
        return {
            "loss": loss,
            "obs_loss": obs_loss,
            "jepa_loss": jepa_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "unlock_loss": unlock_loss,
            "affordance_loss": affordance_loss,
            "done_loss": done_loss,
        }

    @torch.no_grad()
    def prediction_error(self, transition: Transition) -> float:
        self.model.eval()
        obs, action, next_obs, reward, _done, unlock, affordance = batch_to_tensors([[transition]], self.device)
        pred = self.model(obs, action)
        obs_err = F.smooth_l1_loss(pred["next_obs"], next_obs).item()
        reward_err = F.mse_loss(pred["reward"], reward).item()
        unlock_err = F.binary_cross_entropy_with_logits(pred["unlock"], unlock).item()
        affordance_err = F.binary_cross_entropy_with_logits(pred["affordance"], affordance).item()
        return float(obs_err + reward_err + 0.25 * unlock_err + 0.1 * affordance_err)

    @torch.no_grad()
    def predict_skill(self, observation: Observation | np.ndarray, skill: str) -> dict[str, Any]:
        self.model.eval()
        observed_unlocks = encode_unlocks(observation) if isinstance(observation, Observation) else None
        obs_array = encode_observation(observation) if isinstance(observation, Observation) else observation
        obs = torch.tensor(obs_array, dtype=torch.float32, device=self.device).view(1, 1, -1)
        action = torch.tensor(encode_action(skill), dtype=torch.float32, device=self.device).view(1, 1, -1)
        pred = self.model(obs, action)
        policy = torch.softmax(pred["policy"][0, -1], dim=-1)
        unlock = torch.sigmoid(pred["unlock"][0, -1]).detach().cpu().numpy()
        affordance = torch.sigmoid(pred["affordance"][0, -1]).detach().cpu().numpy()
        index = SKILL_NAMES.index(skill) if skill in SKILL_NAMES else 0
        payload = {
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
            "unlock": {name: float(unlock[i]) for i, name in enumerate(UNLOCK_KEYS)},
            "affordance": {name: float(affordance[i]) for i, name in enumerate(SKILL_NAMES)},
            "skill_affordance": float(affordance[index]),
        }
        if observed_unlocks is not None:
            delta = np.maximum(0.0, unlock - observed_unlocks)
            payload["unlock_delta"] = {name: float(delta[i]) for i, name in enumerate(UNLOCK_KEYS)}
            payload["unlock_gain"] = float(delta.sum())
        return payload

    def save(self) -> None:
        torch.save(
            {
                "checkpoint_version": 3,
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "last_loss": self.last_loss,
                "train_step": self.train_step,
                "gamma": self.gamma,
                "skill_names": SKILL_NAMES,
                "model_config": self.model_config,
            },
            self.checkpoint_path,
        )
        self.checkpoint_meta_path.write_text(
            json.dumps(self.checkpoint_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._remember_loaded_checkpoint(self.checkpoint_path)

    def load(self) -> None:
        self.load_from_path(self.checkpoint_path)

    def load_from_path(self, path: Path, *, load_optimizer: bool = True) -> None:
        path = Path(path)
        payload = torch.load(path, map_location=self.device)
        saved_skill_names = tuple(payload.get("skill_names") or ())
        saved_model_config = payload.get("model_config")
        strict_compatible = saved_skill_names == SKILL_NAMES and saved_model_config is not None and dict(saved_model_config) == self.model_config
        if strict_compatible:
            self.model.load_state_dict(payload["model"])
        else:
            loaded, partial = self._load_compatible_model_state(payload.get("model") or {})
            if loaded == 0:
                if saved_skill_names != SKILL_NAMES:
                    raise RuntimeError("checkpoint skill vocabulary does not match current model")
                raise RuntimeError("checkpoint model config does not match current model")
            print(
                f"partially loaded {loaded} world-model tensors from {path}; "
                f"adapted {partial} tensors for changed skills/heads"
            )
        if load_optimizer and strict_compatible and "optim" in payload:
            self.optim.load_state_dict(payload["optim"])
        self.last_loss = float(payload.get("last_loss", 1.0))
        self.train_step = int(payload.get("train_step", 0))
        self.gamma = float(payload.get("gamma", self.gamma))
        self._remember_loaded_checkpoint(path)

    def _load_compatible_model_state(self, saved_state: dict[str, torch.Tensor]) -> tuple[int, int]:
        current = self.model.state_dict()
        loaded = 0
        partial = 0
        for name, saved in saved_state.items():
            if name not in current or not isinstance(saved, torch.Tensor):
                continue
            target = current[name]
            if target.shape == saved.shape:
                target.copy_(saved.to(device=target.device, dtype=target.dtype))
                loaded += 1
                continue
            if target.ndim == saved.ndim:
                slices = tuple(slice(0, min(t_dim, s_dim)) for t_dim, s_dim in zip(target.shape, saved.shape))
                target[slices].copy_(saved.to(device=target.device, dtype=target.dtype)[slices])
                loaded += 1
                partial += 1
        self.model.load_state_dict(current, strict=True)
        return loaded, partial

    def reload_if_changed(self, path: Path | None = None) -> bool:
        path = Path(path or self.checkpoint_path)
        if not path.exists():
            return False
        signature = _checkpoint_signature(path)
        key = str(path.resolve())
        if self._loaded_checkpoint_signatures.get(key) == signature:
            return False
        self.load_from_path(path)
        return True

    def _remember_loaded_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            self._loaded_checkpoint_signatures[str(path.resolve())] = _checkpoint_signature(path)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 3,
            "checkpoint_path": str(self.checkpoint_path),
            "last_loss": self.last_loss,
            "train_step": self.train_step,
            "gamma": self.gamma,
            "skill_count": len(SKILL_NAMES),
            "skill_names": list(SKILL_NAMES),
            "model_config": self.model_config,
        }


def encode_observation(obs: Observation) -> np.ndarray:
    x, y, z = obs.position
    base = [
        obs.health / 20.0,
        obs.food / 20.0,
        np.tanh(x / 256.0),
        np.tanh((y - 64.0) / 64.0),
        np.tanh(z / 256.0),
    ]
    inventory = [np.log1p(obs.inventory.get(name, 0)) / 4.0 for name in INVENTORY_KEYS]
    blocks = [np.log1p(obs.nearby_blocks.get(name, 0)) / 4.0 for name in BLOCK_KEYS]
    return np.asarray(base + inventory + blocks, dtype=np.float32)


def encode_action(skill: str) -> np.ndarray:
    skill = ACTION_ALIASES.get(skill, skill)
    arr = np.zeros(ACTION_DIM, dtype=np.float32)
    if skill in SKILL_NAMES:
        arr[SKILL_NAMES.index(skill)] = 1.0
    return arr


def encode_unlocks(obs: Observation) -> np.ndarray:
    current = progression_snapshot(obs)
    return np.asarray([1.0 if current.get(name, 0) > 0 else 0.0 for name in UNLOCK_KEYS], dtype=np.float32)


def encode_affordances(obs: Observation, target_skill: str | None = None) -> np.ndarray:
    mask = skill_affordance_mask(obs, include_recovery=True)
    values = [1.0 if mask.get(name, False) else 0.0 for name in SKILL_NAMES]
    if target_skill in SKILL_NAMES:
        values[SKILL_NAMES.index(target_skill)] = 1.0
    return np.asarray(values, dtype=np.float32)


def batch_to_tensors(
    sequences: list[list[Transition]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_rows: list[list[np.ndarray]] = []
    action_rows: list[list[np.ndarray]] = []
    next_rows: list[list[np.ndarray]] = []
    reward_rows: list[list[float]] = []
    done_rows: list[list[float]] = []
    unlock_rows: list[list[np.ndarray]] = []
    affordance_rows: list[list[np.ndarray]] = []
    for seq in sequences:
        obs_rows.append([encode_observation(t.observation) for t in seq])
        action_rows.append([encode_action(t.skill) for t in seq])
        next_rows.append([encode_observation(t.next_observation) for t in seq])
        reward_rows.append([float(t.reward) for t in seq])
        done_rows.append([float(t.done) for t in seq])
        unlock_rows.append([encode_unlocks(t.next_observation) for t in seq])
        affordance_rows.append([encode_affordances(t.observation, target_skill=t.skill) for t in seq])
    return (
        torch.tensor(np.asarray(obs_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(action_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(next_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(reward_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(done_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(unlock_rows), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(affordance_rows), dtype=torch.float32, device=device),
    )


def discounted_returns(reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[0], device=reward.device, dtype=reward.dtype)
    for t in range(reward.shape[1] - 1, -1, -1):
        running = reward[:, t] + gamma * running * (1.0 - done[:, t])
        returns[:, t] = running
    return returns


def latent_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.smooth_l1_loss(predicted, target.detach())


def sample_device() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python_rng_probe": random.random(),
    }


def _checkpoint_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size
