from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Iterable

from dataclasses import replace

from mindcraft.skills import SKILLS, WOOD_LOG_ITEMS, WOOD_PLANK_ITEMS, shareable_transfer_count
from mindcraft.schemas import Observation, SkillResult, Transition


PROGRESSION_ITEMS = (
    "logs",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "furnace",
    "stone_pickaxe",
    "coal",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
)

RECOVERY_SKILLS = {"escape_water", "find_crafting_spot", "unstuck_reposition", "move_to_teammate"}
CONTROL_SKILLS = RECOVERY_SKILLS | {"explore_area"}
COMBAT_SKILLS = {"duel_teammate"}

MILESTONE_BONUS = {
    "logs": 1.0,
    "planks": 1.0,
    "stick": 1.0,
    "crafting_table": 5.0,
    "wooden_pickaxe": 8.0,
    "cobblestone": 2.0,
    "furnace": 7.0,
    "stone_pickaxe": 10.0,
    "coal": 4.0,
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

MILESTONE_SKILLS = {
    "logs": "forage_wood",
    "planks": "craft_planks",
    "stick": "craft_sticks",
    "crafting_table": "craft_crafting_table",
    "wooden_pickaxe": "craft_wooden_pickaxe",
    "cobblestone": "mine_stone",
    "stone_pickaxe": "craft_stone_pickaxe",
    "coal": "mine_coal",
    "furnace": "craft_furnace",
    "iron_ore": "mine_iron",
    "iron_ingot": "smelt_iron",
    "iron_pickaxe": "craft_iron_pickaxe",
    "diamond": "mine_diamond",
}

SKILL_MILESTONES = {skill: item for item, skill in MILESTONE_SKILLS.items()}


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
        "coal": inv.get("coal", 0) + inv.get("charcoal", 0),
        "iron_ore": inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0) + inv.get("raw_iron", 0),
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
    if transition.skill in COMBAT_SKILLS and transition.result.data.get("killed"):
        return "combat:kill"
    if transition.skill == "explore_area":
        return "explore"
    if transition.result.success:
        return f"success:{transition.skill}"
    return f"failure:{transition.skill}"


def replay_priority(
    transition: Transition,
    *,
    frontier_items: set[str] | None = None,
    skill_counts: dict[str, int] | None = None,
) -> float:
    delta = transition_delta(transition.observation, transition.next_observation)
    priority = 1.0
    if delta.stage_delta > 0:
        priority += 6.0 + 2.0 * delta.stage_delta
    for milestone in delta.first_milestones:
        priority += MILESTONE_BONUS.get(milestone, 1.0)
    if transition.skill in {"craft_furnace", "craft_stone_pickaxe", "mine_coal", "mine_iron", "smelt_iron", "craft_iron_pickaxe", "mine_diamond"}:
        priority += 3.0
    if transition.skill in COMBAT_SKILLS:
        priority += 6.0 if transition.result.data.get("killed") else 1.5
    if frontier_items:
        touched = set(delta.first_milestones)
        touched.update(name for name, value in delta.item_deltas.items() if value > 0)
        skill_target = SKILL_MILESTONES.get(transition.skill)
        if touched.intersection(frontier_items) or (skill_target and skill_target in frontier_items):
            priority += 8.0
        elif delta.after_stage.index >= max(0, delta.before_stage.index):
            next_targets = set(frontier_items)
            if any(target in {"stone_pickaxe", "coal", "furnace", "iron_ore", "iron_ingot"} for target in next_targets):
                if transition.skill in {"craft_stone_pickaxe", "mine_coal", "craft_furnace", "mine_iron", "smelt_iron"}:
                    priority += 2.5
    if not transition.result.success:
        priority += 0.75
    if transition.skill in CONTROL_SKILLS and not delta.advanced:
        priority *= 0.35
    if skill_counts:
        count = max(1, skill_counts.get(transition.skill, 1))
        median_count = _median_count(skill_counts)
        if transition.skill not in CONTROL_SKILLS and count < median_count:
            priority *= min(3.0, 1.0 + median_count / (count + median_count))
    return max(0.05, priority)


def frontier_items_for_stage(stage: ProgressStage | int) -> tuple[str, ...]:
    index = stage if isinstance(stage, int) else stage.index
    if index <= 0:
        return ("logs", "planks", "stick", "crafting_table")
    if index == 1:
        return ("planks", "stick", "crafting_table", "wooden_pickaxe")
    if index == 2:
        return ("stick", "wooden_pickaxe", "cobblestone")
    if index == 3:
        return ("cobblestone", "stone_pickaxe")
    if index == 4:
        return ("stone_pickaxe", "coal", "furnace")
    if index == 5:
        return ("coal", "furnace", "iron_ore")
    if index == 6:
        return ("coal", "iron_ore", "iron_ingot")
    if index == 7:
        return ("coal", "iron_ingot", "iron_pickaxe")
    if index == 8:
        return ("iron_pickaxe", "diamond")
    return ("diamond",)


def hindsight_relabels(transition: Transition, max_relabels: int = 2) -> list[Transition]:
    if transition.result.data.get("hindsight"):
        return []
    delta = transition_delta(transition.observation, transition.next_observation)
    milestones = list(delta.first_milestones)
    if not milestones:
        milestones = [
            name
            for name, value in sorted(delta.item_deltas.items(), key=lambda item: (MILESTONE_BONUS.get(item[0], 0.0), item[1]), reverse=True)
            if value > 0
        ]
    relabels: list[Transition] = []
    seen: set[str] = set()
    for milestone in milestones:
        skill = MILESTONE_SKILLS.get(milestone)
        if skill is None or skill == transition.skill or skill in seen:
            continue
        if skill not in SKILLS:
            continue
        seen.add(skill)
        reward = max(transition.reward, MILESTONE_BONUS.get(milestone, 1.0) + max(0, delta.stage_delta) * 2.0)
        relabels.append(
            replace(
                transition,
                goal=stage_for_observation(transition.next_observation).goal,
                skill=skill,
                result=SkillResult(
                    skill=skill,
                    success=True,
                    message=f"hindsight relabel: {milestone} via {transition.skill}",
                    data={
                        **transition.result.data,
                        "hindsight": True,
                        "original_skill": transition.skill,
                        "milestone": milestone,
                    },
                    duration_s=transition.result.duration_s,
                ),
                reward=reward,
            )
        )
        if len(relabels) >= max_relabels:
            break
    return relabels


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
    coal = current["coal"]
    iron_ore = current["iron_ore"]
    iron_ingot = current["iron_ingot"]

    if water_pressure(obs) >= 64:
        return ["escape_water", "unstuck_reposition"]
    if current["diamond"] > 0:
        return ["share_supplies", "explore_area"]
    if iron_pickaxe > 0:
        return ["mine_diamond", "explore_area"]
    if iron_ingot >= 3:
        if iron_pickaxe <= 0:
            return ["complete_iron_pickaxe", "craft_iron_pickaxe"]
        if sticks < 2:
            return ["craft_sticks", "craft_planks", "forage_wood"]
        if table <= 0:
            return _table_setup_order(inv, planks, obs)
        return ["craft_iron_pickaxe"]
    if iron_ore > 0:
        if furnace <= 0:
            return _furnace_order(inv, obs, cobblestone, table)
        if not has_fuel(obs):
            if nearby_wood(obs):
                return ["forage_wood", "craft_planks"]
            return ["explore_area"]
        return ["smelt_iron"]
    if stone_pickaxe > 0:
        if furnace <= 0:
            return _furnace_order(inv, obs, cobblestone, table)
        coal_visible = obs.nearby_blocks.get("coal_ore", 0) or obs.nearby_blocks.get("deepslate_coal_ore", 0)
        iron_visible = obs.nearby_blocks.get("iron_ore", 0) or obs.nearby_blocks.get("deepslate_iron_ore", 0)
        if coal <= 0 and coal_visible and _line_of_sight_in(obs, {"coal_ore", "deepslate_coal_ore"}):
            return ["mine_coal", "mine_iron", "explore_area"]
        if iron_visible and _line_of_sight_in(obs, {"iron_ore", "deepslate_iron_ore"}):
            return ["mine_iron", "explore_area"]
        return ["explore_area"]
    if cobblestone >= 3 and sticks >= 2:
        if table <= 0:
            return _table_setup_order(inv, planks, obs)
        return ["craft_stone_pickaxe"]
    if wooden_pickaxe > 0:
        if cobblestone < 3:
            if obs.nearby_blocks.get("stone", 0) or obs.nearby_blocks.get("deepslate", 0):
                return ["mine_stone"]
            return ["explore_area"]
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


def skill_preconditions_ok(name: str, obs: Observation) -> bool:
    inv = obs.inventory
    planks = count_planks(inv)
    logs = count_logs(inv)
    table = has_table_access(obs)
    pickaxes = inv.get("wooden_pickaxe", 0) + inv.get("stone_pickaxe", 0) + inv.get("iron_pickaxe", 0)
    if name == "craft_planks":
        return logs > 0
    if name == "craft_sticks":
        return planks >= 2
    if name == "craft_crafting_table":
        return planks >= 4 and inv.get("crafting_table", 0) == 0 and obs.nearby_blocks.get("crafting_table", 0) == 0
    if name == "forage_wood":
        return nearby_wood(obs)
    if name == "place_crafting_table":
        return inv.get("crafting_table", 0) > 0
    if name == "craft_wooden_pickaxe":
        return planks >= 3 and inv.get("stick", 0) >= 2 and table
    if name == "mine_stone":
        return pickaxes > 0
    if name == "craft_stone_pickaxe":
        return inv.get("cobblestone", 0) >= 3 and inv.get("stick", 0) >= 2 and table
    if name == "mine_coal":
        return pickaxes > 0
    if name == "mine_iron":
        return inv.get("stone_pickaxe", 0) + inv.get("iron_pickaxe", 0) > 0
    if name == "craft_furnace":
        return inv.get("cobblestone", 0) >= 8 and inv.get("furnace", 0) == 0 and table
    if name == "smelt_iron":
        iron_inputs = inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0) + inv.get("raw_iron", 0)
        return iron_inputs > 0 and inv.get("furnace", 0) > 0 and has_fuel(obs)
    if name == "craft_iron_pickaxe":
        return inv.get("iron_ingot", 0) >= 3 and inv.get("stick", 0) >= 2 and inv.get("iron_pickaxe", 0) == 0 and table
    if name == "complete_iron_pickaxe":
        return inv.get("iron_ingot", 0) >= 3 and inv.get("iron_pickaxe", 0) == 0
    if name == "mine_diamond":
        return inv.get("iron_pickaxe", 0) > 0
    if name == "share_supplies":
        if obs.nearby_entities.get("player", 0) <= 0:
            return False
        if inv.get("iron_pickaxe", 0) > 0 and inv.get("diamond", 0) == 0:
            return False
        return any(shareable_transfer_count(item, count) > 0 for item, count in inv.items())
    if name == "duel_teammate":
        if inv.get("diamond", 0) > 0 or inv.get("iron_pickaxe", 0) > 0:
            return False
        return obs.nearby_entities.get("player", 0) > 0
    if name == "escape_water":
        return water_pressure(obs) >= 24
    if name == "find_crafting_spot":
        return inv.get("crafting_table", 0) > 0 or obs.nearby_blocks.get("crafting_table", 0) > 0
    if name == "unstuck_reposition":
        return True
    if name == "move_to_teammate":
        return obs.nearby_entities.get("player", 0) > 0
    return name in SKILLS


def skill_affordance_mask(obs: Observation, *, include_recovery: bool = True) -> dict[str, bool]:
    return {
        name: skill_preconditions_ok(name, obs) and (include_recovery or name not in RECOVERY_SKILLS)
        for name in SKILLS
    }


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
        if obs.nearby_blocks.get("stone", 0) or obs.nearby_blocks.get("deepslate", 0):
            return ["mine_stone"]
        return ["explore_area"]
    if table <= 0:
        return _table_setup_order(inv, count_planks(inv), obs)
    if not has_fuel(obs) and (obs.nearby_blocks.get("coal_ore", 0) or obs.nearby_blocks.get("deepslate_coal_ore", 0)):
        return ["mine_coal", "craft_furnace"]
    return ["craft_furnace"]


def _line_of_sight_in(obs: Observation, names: set[str]) -> bool:
    return obs.line_of_sight in names


def _median_count(counts: dict[str, int]) -> float:
    values = sorted(value for value in counts.values() if value > 0)
    if not values:
        return 1.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return 0.5 * (values[mid - 1] + values[mid])
