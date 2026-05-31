from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Iterable

from mindcraft.skills import WOOD_LOG_ITEMS, WOOD_PLANK_ITEMS
from mindcraft.schemas import Observation, Transition


PROGRESSION_ITEMS = (
    "logs",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "furnace",
    "stone_pickaxe",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
)

RECOVERY_SKILLS = {"escape_water", "find_crafting_spot", "unstuck_reposition", "move_to_teammate"}
CONTROL_SKILLS = RECOVERY_SKILLS | {"explore_area"}

MILESTONE_BONUS = {
    "logs": 1.0,
    "planks": 1.0,
    "stick": 1.0,
    "crafting_table": 5.0,
    "wooden_pickaxe": 8.0,
    "cobblestone": 2.0,
    "furnace": 7.0,
    "stone_pickaxe": 10.0,
    "iron_ore": 12.0,
    "iron_ingot": 14.0,
    "iron_pickaxe": 20.0,
    "diamond": 80.0,
}


@dataclass(frozen=True, slots=True)
class ProgressStage:
    index: int
    name: str
    goal: str


@dataclass(frozen=True, slots=True)
class ProgressDelta:
    before_stage: ProgressStage
    after_stage: ProgressStage
    item_deltas: dict[str, int]
    first_milestones: tuple[str, ...]
    distance_moved: float
    water_delta: int

    @property
    def stage_delta(self) -> int:
        return self.after_stage.index - self.before_stage.index

    @property
    def advanced(self) -> bool:
        return self.stage_delta > 0 or bool(self.first_milestones)


STAGES = (
    ProgressStage(0, "bootstrap", "bootstrap"),
    ProgressStage(1, "wood", "wood"),
    ProgressStage(2, "table_setup", "tools"),
    ProgressStage(3, "wooden_tool", "stone"),
    ProgressStage(4, "stone_stockpile", "stone"),
    ProgressStage(5, "stone_tool", "iron"),
    ProgressStage(6, "furnace_ready", "iron"),
    ProgressStage(7, "iron_ore", "iron"),
    ProgressStage(8, "iron_ingots", "diamond"),
    ProgressStage(9, "iron_tool", "diamond"),
    ProgressStage(10, "diamond", "culture"),
)


def snapshot(obs: Observation) -> dict[str, int]:
    inv = obs.inventory
    nearby = obs.nearby_blocks
    return {
        "logs": count_logs(inv),
        "planks": count_planks(inv),
        "stick": inv.get("stick", 0),
        "crafting_table": inv.get("crafting_table", 0) + nearby.get("crafting_table", 0),
        "wooden_pickaxe": inv.get("wooden_pickaxe", 0),
        "cobblestone": inv.get("cobblestone", 0),
        "furnace": inv.get("furnace", 0) + nearby.get("furnace", 0),
        "stone_pickaxe": inv.get("stone_pickaxe", 0),
        "iron_ore": inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0),
        "iron_ingot": inv.get("iron_ingot", 0),
        "iron_pickaxe": inv.get("iron_pickaxe", 0),
        "diamond": inv.get("diamond", 0),
    }


def stage_for_observation(obs: Observation) -> ProgressStage:
    current = snapshot(obs)
    if current["diamond"] > 0:
        return STAGES[10]
    if current["iron_pickaxe"] > 0:
        return STAGES[9]
    if current["iron_ingot"] >= 3:
        return STAGES[8]
    if current["iron_ore"] > 0:
        return STAGES[7]
    if current["stone_pickaxe"] > 0 and current["furnace"] > 0:
        return STAGES[6]
    if current["stone_pickaxe"] > 0:
        return STAGES[5]
    if current["wooden_pickaxe"] > 0 and current["cobblestone"] >= 3:
        return STAGES[4]
    if current["wooden_pickaxe"] > 0:
        return STAGES[3]
    if current["crafting_table"] > 0:
        return STAGES[2]
    if current["logs"] > 0 or current["planks"] > 0:
        return STAGES[1]
    return STAGES[0]


def team_stage(observations: Iterable[Observation]) -> ProgressStage:
    stages = [stage_for_observation(obs) for obs in observations]
    return max(stages, key=lambda stage: stage.index) if stages else STAGES[0]


def goal_for_observation(obs: Observation) -> str:
    return stage_for_observation(obs).goal


def transition_delta(before: Observation, after: Observation) -> ProgressDelta:
    before_snapshot = snapshot(before)
    after_snapshot = snapshot(after)
    item_deltas = {
        name: after_snapshot.get(name, 0) - before_snapshot.get(name, 0)
        for name in PROGRESSION_ITEMS
    }
    first = tuple(
        name
        for name in PROGRESSION_ITEMS
        if before_snapshot.get(name, 0) <= 0 and after_snapshot.get(name, 0) > 0
    )
    return ProgressDelta(
        before_stage=stage_for_observation(before),
        after_stage=stage_for_observation(after),
        item_deltas=item_deltas,
        first_milestones=first,
        distance_moved=dist(before.position, after.position),
        water_delta=water_pressure(after) - water_pressure(before),
    )


def transition_event_bucket(transition: Transition) -> str:
    delta = transition_delta(transition.observation, transition.next_observation)
    if delta.first_milestones:
        return f"milestone:{delta.first_milestones[-1]}"
    if delta.stage_delta > 0:
        return f"stage:{delta.after_stage.name}"
    if transition.skill in RECOVERY_SKILLS:
        return f"recovery:{transition.skill}"
    if transition.skill == "explore_area":
        return "explore"
    if transition.result.success:
        return f"success:{transition.skill}"
    return f"failure:{transition.skill}"


def replay_priority(transition: Transition) -> float:
    delta = transition_delta(transition.observation, transition.next_observation)
    priority = 1.0
    if delta.stage_delta > 0:
        priority += 6.0 + 2.0 * delta.stage_delta
    for milestone in delta.first_milestones:
        priority += MILESTONE_BONUS.get(milestone, 1.0)
    if transition.skill in {"craft_furnace", "craft_stone_pickaxe", "mine_iron", "smelt_iron", "craft_iron_pickaxe", "mine_diamond"}:
        priority += 3.0
    if not transition.result.success:
        priority += 0.75
    if transition.skill in CONTROL_SKILLS and not delta.advanced:
        priority *= 0.35
    return max(0.05, priority)


def curriculum_skill_order(obs: Observation) -> list[str]:
    inv = obs.inventory
    current = snapshot(obs)
    logs = current["logs"]
    planks = current["planks"]
    sticks = current["stick"]
    table = current["crafting_table"]
    wooden_pickaxe = current["wooden_pickaxe"]
    stone_pickaxe = current["stone_pickaxe"]
    iron_pickaxe = current["iron_pickaxe"]
    cobblestone = current["cobblestone"]
    furnace = current["furnace"]
    iron_ore = current["iron_ore"]
    iron_ingot = current["iron_ingot"]

    if water_pressure(obs) >= 64:
        return ["escape_water", "unstuck_reposition"]
    if current["diamond"] > 0:
        return ["share_supplies", "explore_area"]
    if iron_pickaxe > 0:
        return ["mine_diamond", "explore_area"]
    if iron_ingot >= 3:
        if sticks < 2:
            return ["craft_sticks", "craft_planks", "forage_wood"]
        if table <= 0:
            return _table_setup_order(inv, planks, obs)
        return ["craft_iron_pickaxe"]
    if iron_ore > 0:
        if furnace <= 0:
            return _furnace_order(inv, obs, cobblestone, table)
        if not has_fuel(obs):
            return ["forage_wood", "craft_planks"]
        return ["smelt_iron"]
    if stone_pickaxe > 0:
        if furnace <= 0:
            return _furnace_order(inv, obs, cobblestone, table)
        return ["mine_iron", "explore_area"]
    if cobblestone >= 3 and sticks >= 2:
        if table <= 0:
            return _table_setup_order(inv, planks, obs)
        return ["craft_stone_pickaxe"]
    if wooden_pickaxe > 0:
        if cobblestone < 3:
            return ["mine_stone"]
        if sticks < 2:
            return ["craft_sticks", "craft_planks", "forage_wood"]
        if table <= 0:
            return _table_setup_order(inv, planks, obs)
        return ["craft_stone_pickaxe", "mine_stone"]
    if table <= 0:
        return _table_setup_order(inv, planks, obs)
    if sticks < 2:
        if planks >= 2:
            return ["craft_sticks"]
        return _wood_order(logs, planks, obs)
    if planks < 3:
        return _wood_order(logs, planks, obs)
    return ["craft_wooden_pickaxe"]


def progress_summary(before: Observation, after: Observation) -> dict[str, object]:
    delta = transition_delta(before, after)
    return {
        "stage_before": delta.before_stage.name,
        "stage_after": delta.after_stage.name,
        "stage_index_before": delta.before_stage.index,
        "stage_index_after": delta.after_stage.index,
        "stage_delta": delta.stage_delta,
        "first_milestones": list(delta.first_milestones),
        "item_deltas": {name: value for name, value in delta.item_deltas.items() if value},
        "distance_moved": delta.distance_moved,
        "water_delta": delta.water_delta,
    }


def count_logs(inv: dict[str, int]) -> int:
    return sum(count for name, count in inv.items() if name in WOOD_LOG_ITEMS or name.endswith("_log") or name.endswith("_stem"))


def count_planks(inv: dict[str, int]) -> int:
    return sum(count for name, count in inv.items() if name in WOOD_PLANK_ITEMS or name.endswith("_planks"))


def has_table_access(obs: Observation) -> bool:
    return obs.inventory.get("crafting_table", 0) > 0 or obs.nearby_blocks.get("crafting_table", 0) > 0


def has_fuel(obs: Observation) -> bool:
    inv = obs.inventory
    return (
        inv.get("coal", 0) > 0
        or inv.get("charcoal", 0) > 0
        or count_logs(inv) > 0
        or count_planks(inv) > 0
    )


def nearby_wood(obs: Observation) -> bool:
    return any(obs.nearby_blocks.get(name, 0) > 0 for name in WOOD_LOG_ITEMS) or any(
        name.endswith("_log") or name.endswith("_stem")
        for name, count in obs.nearby_blocks.items()
        if count > 0
    )


def water_pressure(obs: Observation) -> int:
    return sum(count for name, count in obs.nearby_blocks.items() if "water" in name or "lava" in name)


def _wood_order(logs: int, planks: int, obs: Observation) -> list[str]:
    if logs > 0 and planks < 12:
        return ["craft_planks", "forage_wood"]
    if nearby_wood(obs):
        return ["forage_wood"]
    return ["explore_area"]


def _table_setup_order(inv: dict[str, int], planks: int, obs: Observation | None = None) -> list[str]:
    if inv.get("crafting_table", 0) > 0:
        return ["place_crafting_table", "find_crafting_spot"]
    if planks >= 4:
        return ["craft_crafting_table"]
    if obs is not None:
        return _wood_order(count_logs(inv), planks, obs)
    return ["craft_planks", "forage_wood", "explore_area"]


def _furnace_order(inv: dict[str, int], obs: Observation, cobblestone: int, table: int) -> list[str]:
    if cobblestone < 8:
        return ["mine_stone"]
    if table <= 0:
        return _table_setup_order(inv, count_planks(inv), obs)
    return ["craft_furnace"]
