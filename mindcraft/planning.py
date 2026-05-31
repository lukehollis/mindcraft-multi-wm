from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindcraft.skill_library import SkillLibrary
from mindcraft.schemas import Observation
from mindcraft.world_model import WorldModelTrainer, encode_observation


@dataclass(slots=True)
class MCTSPlan:
    skill: str
    visits: int
    value: float
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class _Edge:
    skill: str
    prior: float
    raw_reward: float
    reward: float
    value_hint: float
    uncertainty: float
    child: "_Node"
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return self.reward + self.value_hint
        return self.value_sum / self.visits


@dataclass(slots=True)
class _Node:
    state: np.ndarray
    children: dict[str, _Edge] = field(default_factory=dict)


class ModelMCTSPlanner:
    def __init__(
        self,
        world_model: WorldModelTrainer,
        simulations: int = 32,
        depth: int = 3,
        c_puct: float = 1.35,
        uncertainty_weight: float = 0.35,
        gamma: float = 0.97,
    ):
        self.world_model = world_model
        self.simulations = simulations
        self.depth = depth
        self.c_puct = c_puct
        self.uncertainty_weight = uncertainty_weight
        self.gamma = gamma

    def select(
        self,
        observation: Observation,
        candidates: list[str],
        library: SkillLibrary,
        rng: random.Random,
        agent_id: str | None = None,
    ) -> MCTSPlan:
        if not candidates:
            raise ValueError("MCTS planner needs at least one candidate skill")
        root = _Node(encode_observation(observation))
        self._expand(root, candidates, library, agent_id)
        for _ in range(max(1, self.simulations)):
            self._simulate(root, candidates, library, rng, self.depth, agent_id)
        best = max(root.children.values(), key=lambda edge: (edge.visits, edge.q_value, edge.prior))
        diagnostics = {
            name: {
                "visits": edge.visits,
                "q": edge.q_value,
                "prior": edge.prior,
                "reward": edge.raw_reward,
                "penalized_reward": edge.reward,
                "uncertainty": edge.uncertainty,
            }
            for name, edge in sorted(root.children.items())
        }
        return MCTSPlan(skill=best.skill, visits=best.visits, value=best.q_value, diagnostics=diagnostics)

    def _simulate(
        self,
        root: _Node,
        candidates: list[str],
        library: SkillLibrary,
        rng: random.Random,
        depth: int,
        agent_id: str | None,
    ) -> float:
        node = root
        path: list[_Edge] = []
        for remaining_depth in range(depth):
            if not node.children:
                self._expand(node, candidates, library, agent_id)
                return self._leaf_value(node)
            edge = self._select_edge(node, rng)
            path.append(edge)
            node = edge.child
            if not node.children and remaining_depth + 1 < depth:
                self._expand(node, candidates, library, agent_id)
                break
        value = self._leaf_value(node)
        for edge in reversed(path):
            value = edge.reward + self.gamma * value
            edge.visits += 1
            edge.value_sum += value
        return value

    def _expand(self, node: _Node, candidates: list[str], library: SkillLibrary, agent_id: str | None) -> None:
        if node.children:
            return
        predictions = {skill: self.world_model.predict_skill(node.state, skill) for skill in candidates}
        prior_sum = sum(max(1.0e-6, float(pred["prior"])) for pred in predictions.values())
        for skill, pred in predictions.items():
            stat = library.stats_for(skill, agent_id)
            learned_bias = 0.0 if stat is None else math.tanh(stat.q_value / 20.0) * 0.05
            prior = max(1.0e-6, float(pred["prior"])) / max(1.0e-6, prior_sum)
            prior = max(1.0e-6, prior + learned_bias)
            reward_uncertainty = max(0.0, float(pred["reward_uncertainty"]))
            value_uncertainty = max(0.0, float(pred["value_uncertainty"]))
            uncertainty = reward_uncertainty + self.gamma * value_uncertainty
            node.children[skill] = _Edge(
                skill=skill,
                prior=prior,
                raw_reward=float(pred["reward"]),
                reward=float(pred["reward"]) - self.uncertainty_weight * reward_uncertainty,
                value_hint=float(pred["value"]) - self.uncertainty_weight * value_uncertainty,
                uncertainty=uncertainty,
                child=_Node(np.asarray(pred["next_obs"], dtype=np.float32)),
            )
        norm = sum(edge.prior for edge in node.children.values())
        for edge in node.children.values():
            edge.prior /= max(1.0e-6, norm)

    def _select_edge(self, node: _Node, rng: random.Random) -> _Edge:
        total_visits = sum(edge.visits for edge in node.children.values())
        scored: list[tuple[float, float, _Edge]] = []
        for edge in node.children.values():
            exploration = self.c_puct * edge.prior * math.sqrt(total_visits + 1) / (edge.visits + 1)
            jitter = rng.random() * 1.0e-6
            scored.append((edge.q_value + exploration, jitter, edge))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _leaf_value(self, node: _Node) -> float:
        if not node.children:
            return 0.0
        return max(edge.reward + self.gamma * edge.value_hint for edge in node.children.values())
