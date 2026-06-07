from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mindcraft.skills import SKILLS
from mindcraft.schemas import Observation
from mindcraft.progression import (
    count_logs,
    count_planks,
    curriculum_skill_order,
    has_table_access,
    nearby_wood,
    skill_preconditions_ok,
    water_pressure,
)

SKILL_ALIASES = {
    "scout_area": "explore_area",
}
RECOVERY_SKILLS = {"escape_water", "find_crafting_spot", "unstuck_reposition", "move_to_teammate"}
COMBAT_SKILLS = {"duel_teammate"}


@dataclass(slots=True)
class SkillStats:
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


class SkillLibrary:
    def __init__(self, path: Path, alpha: float = 0.18):
        self.path = path
        self.alpha = alpha
        self.stats: dict[str, SkillStats] = {name: SkillStats() for name in SKILLS}
        self.agent_stats: dict[str, dict[str, SkillStats]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.load()

    @property
    def total_attempts(self) -> int:
        return sum(stat.attempts for stat in self.stats.values())

    def candidates(self, agent_id: str, goal: str, observation: Observation) -> list[str]:
        names = []
        for name, skill in SKILLS.items():
            if name in COMBAT_SKILLS and goal not in {"stone", "iron", "diamond", "culture", "combat"}:
                continue
            if goal not in skill.goal_tags and not set(skill.goal_tags).intersection(_goal_family(goal)):
                continue
            if self._preconditions_ok(name, observation):
                names.append(name)
        if not names:
            names = [
                name
                for name in SKILLS
                if name not in RECOVERY_SKILLS
                and (name != "share_supplies" or goal in {"diamond", "culture"})
                and (name not in COMBAT_SKILLS or goal in {"stone", "iron", "diamond", "culture", "combat"})
                and self._preconditions_ok(name, observation)
            ]
        return names or list(SKILLS)

    def select(
        self,
        agent_id: str,
        goal: str,
        observation: Observation,
        rng: random.Random,
        epsilon: float,
        ucb_c: float,
        curiosity_weight: float,
    ) -> str:
        candidates = self.candidates(agent_id, goal, observation)
        if rng.random() < epsilon:
            return rng.choice(candidates)
        scores = self.score_candidates(agent_id, goal, observation, ucb_c, curiosity_weight)
        return max(candidates, key=lambda name: (scores.get(name, float("-inf")), name))

    def score_candidates(
        self,
        agent_id: str,
        goal: str,
        observation: Observation,
        ucb_c: float,
        curiosity_weight: float,
    ) -> dict[str, float]:
        candidates = self.candidates(agent_id, goal, observation)
        global_total = max(1, self.total_attempts)
        agent_table = self._agent_table(agent_id)
        agent_total = max(1, sum(stat.attempts for stat in agent_table.values()))
        scores: dict[str, float] = {}
        for name in candidates:
            global_stat = self.stats[name]
            agent_stat = agent_table[name]
            global_ucb = 0.35 * ucb_c * math.sqrt(math.log(global_total + 1) / (global_stat.attempts + 1))
            agent_ucb = ucb_c * math.sqrt(math.log(agent_total + 1) / (agent_stat.attempts + 1))
            q_value = 0.35 * global_stat.q_value + 0.65 * agent_stat.q_value
            prediction_error = 0.35 * global_stat.prediction_error + 0.65 * agent_stat.prediction_error
            curiosity = curiosity_weight * prediction_error
            precondition_bonus = self._inventory_affordance_bonus(name, observation)
            scores[name] = q_value + global_ucb + agent_ucb + curiosity + precondition_bonus
        return scores

    def affordance_bonuses(self, agent_id: str, goal: str, observation: Observation) -> dict[str, float]:
        return {
            name: self._inventory_affordance_bonus(name, observation)
            for name in self.candidates(agent_id, goal, observation)
        }

    def curriculum_candidates(self, agent_id: str, goal: str, observation: Observation) -> list[str]:
        available = {
            name
            for name in self.candidates(agent_id, goal, observation)
            if self._preconditions_ok(name, observation)
        }
        return [name for name in curriculum_skill_order(observation) if name in available]

    def update(
        self,
        skill: str,
        reward: float,
        success: bool,
        episode: int,
        step: int,
        prediction_error: float | None = None,
        note: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        skill = _canonical_skill(skill)
        stat = self.stats.setdefault(skill, SkillStats())
        self._update_stat(stat, reward, success, episode, step, prediction_error, note)
        if agent_id is not None:
            agent_stat = self._agent_table(agent_id).setdefault(skill, SkillStats())
            self._update_stat(agent_stat, reward, success, episode, step, prediction_error, note)

    def _update_stat(
        self,
        stat: SkillStats,
        reward: float,
        success: bool,
        episode: int,
        step: int,
        prediction_error: float | None,
        note: str | None,
    ) -> None:
        stat.attempts += 1
        stat.successes += int(success)
        stat.total_reward += reward
        stat.q_value = (1.0 - self.alpha) * stat.q_value + self.alpha * reward
        if prediction_error is not None:
            stat.prediction_error = 0.9 * stat.prediction_error + 0.1 * float(prediction_error)
        else:
            stat.prediction_error *= 0.995
        stat.last_episode = episode
        stat.last_step = step
        if note:
            stat.notes = (stat.notes + [note])[-12:]

    def stats_for(self, skill: str, agent_id: str | None = None) -> SkillStats | None:
        skill = _canonical_skill(skill)
        if agent_id is None:
            return self.stats.get(skill)
        agent_stat = self.agent_stats.get(agent_id, {}).get(skill)
        return agent_stat if agent_stat is not None and agent_stat.attempts > 0 else self.stats.get(skill)

    def save(self) -> None:
        payload = {
            "version": 2,
            "skills": {name: asdict(stat) for name, stat in self.stats.items()},
            "agents": {
                agent_id: {name: asdict(stat) for name, stat in stats.items() if stat.attempts > 0}
                for agent_id, stats in sorted(self.agent_stats.items())
            },
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if "skills" in payload:
            skill_payload = payload.get("skills") or {}
            agent_payload = payload.get("agents") or {}
        else:
            skill_payload = payload
            agent_payload = {}
        for name, stat in skill_payload.items():
            canonical = _canonical_skill(name)
            if canonical in SKILLS:
                self.stats[canonical] = _merge_stats(self.stats.get(canonical, SkillStats()), SkillStats(**stat))
        for agent_id, stats in agent_payload.items():
            table = self._agent_table(str(agent_id))
            for name, stat in stats.items():
                canonical = _canonical_skill(name)
                if canonical in SKILLS:
                    table[canonical] = _merge_stats(table.get(canonical, SkillStats()), SkillStats(**stat))

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            "skills": self._summary_table(self.stats),
            "agents": {
                agent_id: self._summary_table(stats, include_empty=False)
                for agent_id, stats in sorted(self.agent_stats.items())
            },
        }

    def _summary_table(self, stats: dict[str, SkillStats], include_empty: bool = True) -> dict[str, dict[str, float]]:
        return {
            name: {
                "attempts": current.attempts,
                "success_rate": current.success_rate,
                "q_value": current.q_value,
                "prediction_error": current.prediction_error,
            }
            for name in SKILLS
            for current in [stats.get(name, SkillStats())]
            if include_empty or current.attempts > 0
        }

    def _agent_table(self, agent_id: str) -> dict[str, SkillStats]:
        return self.agent_stats.setdefault(agent_id, {name: SkillStats() for name in SKILLS})

    def _preconditions_ok(self, name: str, obs: Observation) -> bool:
        return skill_preconditions_ok(name, obs)

    def _inventory_affordance_bonus(self, name: str, obs: Observation) -> float:
        if not self._preconditions_ok(name, obs):
            return -10.0
        inv = obs.inventory
        logs = count_logs(inv)
        planks = count_planks(inv)
        pickaxes = inv.get("wooden_pickaxe", 0) + inv.get("stone_pickaxe", 0) + inv.get("iron_pickaxe", 0)
        if name == "craft_planks" and logs > 0 and planks < 16:
            return 5.0
        if name == "craft_sticks" and planks >= 2 and inv.get("stick", 0) < 6:
            return 30.0 if inv.get("iron_ingot", 0) >= 3 and inv.get("stick", 0) < 2 else 4.0
        if name == "craft_crafting_table" and planks >= 4 and inv.get("crafting_table", 0) == 0:
            return 12.0
        if name == "place_crafting_table" and inv.get("crafting_table", 0) > 0:
            return 1.5
        if name == "craft_wooden_pickaxe" and planks >= 3 and inv.get("stick", 0) >= 2 and pickaxes == 0:
            return 7.0
        if name == "craft_stone_pickaxe" and inv.get("cobblestone", 0) >= 3 and inv.get("stick", 0) >= 2 and inv.get("stone_pickaxe", 0) == 0:
            return 8.0
        if name == "craft_furnace" and inv.get("cobblestone", 0) >= 8 and inv.get("furnace", 0) == 0 and has_table_access(obs):
            return 18.0
        if name == "smelt_iron" and _raw_or_ore_iron(inv) > 0 and inv.get("furnace", 0) > 0:
            return 28.0 if inv.get("iron_pickaxe", 0) == 0 else -20.0
        if name == "craft_iron_pickaxe" and inv.get("iron_ingot", 0) >= 3 and inv.get("stick", 0) >= 2:
            return 35.0
        if name == "complete_iron_pickaxe" and inv.get("iron_ingot", 0) >= 3 and inv.get("iron_pickaxe", 0) == 0:
            return 45.0
        if name == "forage_wood" and nearby_wood(obs):
            return 0.4 if logs + planks < 12 else -1.0
        if name == "mine_stone" and obs.nearby_blocks.get("stone", 0):
            return 8.0 if pickaxes and inv.get("cobblestone", 0) < 16 else 0.35
        if name == "mine_coal" and (obs.nearby_blocks.get("coal_ore", 0) or obs.nearby_blocks.get("deepslate_coal_ore", 0)):
            return 10.0 if pickaxes and inv.get("coal", 0) < 8 else 0.5
        if name == "mine_iron" and (obs.nearby_blocks.get("iron_ore", 0) or obs.nearby_blocks.get("deepslate_iron_ore", 0)):
            if inv.get("iron_pickaxe", 0) > 0:
                return -20.0
            return 9.0 if _raw_or_ore_iron(inv) < 3 else -2.0
        if name == "mine_diamond" and (obs.nearby_blocks.get("diamond_ore", 0) or obs.nearby_blocks.get("deepslate_diamond_ore", 0)):
            return 100.0
        if name == "mine_diamond" and inv.get("iron_pickaxe", 0) > 0:
            return 12.0
        if name == "share_supplies":
            if inv.get("stick", 0) >= 2 or inv.get("iron_ingot", 0) >= 3 or _raw_or_ore_iron(inv) >= 3:
                return 10.0
            return 1.0
        if name == "duel_teammate":
            nearby_players = obs.nearby_entities.get("player", 0)
            if nearby_players <= 0:
                return -10.0
            carried_value = sum(count * _item_value(item) for item, count in inv.items())
            if carried_value >= 50.0:
                return -6.0
            return 1.5 if carried_value <= 8.0 else 0.5
        if name == "escape_water":
            return 12.0 if water_pressure(obs) >= 48 else 3.0
        if name == "find_crafting_spot":
            return 6.0 if inv.get("crafting_table", 0) > 0 else 1.0
        if name == "unstuck_reposition":
            return -2.0
        if name == "move_to_teammate":
            return 0.2
        return 0.0


def _goal_family(goal: str) -> set[str]:
    families = {
        "bootstrap": {"bootstrap", "wood", "tools", "explore"},
        "wood": {"bootstrap", "wood", "tools"},
        "tools": {"tools", "wood", "stone"},
        "stone": {"stone", "tools", "iron", "combat"},
        "iron": {"iron", "combat"},
        "diamond": {"diamond", "iron", "tools", "wood", "explore", "combat"},
        "culture": {"culture", "diamond", "combat"},
        "combat": {"combat", "culture"},
    }
    return families.get(goal, {goal})


def _item_value(item: str) -> float:
    if item.endswith("_log") or item.endswith("_stem"):
        return 1.0
    if item.endswith("_planks"):
        return 0.25
    return {
        "stick": 0.15,
        "crafting_table": 2.0,
        "wooden_pickaxe": 4.0,
        "cobblestone": 0.4,
        "stone_pickaxe": 6.0,
        "coal": 1.0,
        "iron_ore": 8.0,
        "raw_iron": 8.0,
        "furnace": 3.0,
        "iron_ingot": 10.0,
        "iron_pickaxe": 16.0,
        "diamond": 60.0,
    }.get(item, 0.0)


def _raw_or_ore_iron(inv: dict[str, int]) -> int:
    return inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0) + inv.get("raw_iron", 0)


def _canonical_skill(name: str) -> str:
    return SKILL_ALIASES.get(name, name)


def _merge_stats(left: SkillStats, right: SkillStats) -> SkillStats:
    if left.attempts == 0:
        return right
    if right.attempts == 0:
        return left
    return SkillStats(
        attempts=left.attempts + right.attempts,
        successes=left.successes + right.successes,
        total_reward=left.total_reward + right.total_reward,
        q_value=right.q_value,
        prediction_error=right.prediction_error,
        last_episode=max(left.last_episode, right.last_episode),
        last_step=max(left.last_step, right.last_step),
        notes=(left.notes + right.notes)[-12:],
    )
