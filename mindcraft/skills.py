from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


OVERWORLD_WOOD_TYPES = (
    "oak",
    "birch",
    "spruce",
    "jungle",
    "acacia",
    "dark_oak",
    "mangrove",
    "cherry",
)
WOOD_LOG_ITEMS = tuple(f"{name}_log" for name in OVERWORLD_WOOD_TYPES) + ("crimson_stem", "warped_stem")
WOOD_PLANK_ITEMS = tuple(f"{name}_planks" for name in OVERWORLD_WOOD_TYPES) + (
    "crimson_planks",
    "warped_planks",
    "bamboo_planks",
)
WOOD_SOURCE_BLOCKS = WOOD_LOG_ITEMS
SHARE_ITEM_PRIORITY = (
    "diamond",
    "iron_ingot",
    "iron_ore",
    "iron_pickaxe",
    "stone_pickaxe",
    "furnace",
    "cobblestone",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "coal",
    *WOOD_PLANK_ITEMS,
    *WOOD_LOG_ITEMS,
)
SHARE_BATCH_SIZE = {
    "iron_ingot": 3,
    "iron_ore": 3,
    "stick": 2,
    "cobblestone": 8,
}


def shareable_transfer_count(item: str, count: int) -> int:
    if item == "diamond":
        return count
    if item in {"iron_ingot", "iron_ore"}:
        return count if count >= 3 else 0
    if item in {"iron_pickaxe", "stone_pickaxe", "furnace", "crafting_table", "wooden_pickaxe"}:
        return max(0, count - 1)
    if item == "cobblestone":
        return max(0, count - 16)
    if item == "stick":
        return count if count >= 2 else 0
    if item == "coal":
        return max(0, count - 4)
    if item in WOOD_LOG_ITEMS or item in WOOD_PLANK_ITEMS:
        return max(0, count - 16)
    return 0


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    goal_tags: tuple[str, ...]
    default_args: dict[str, Any] = field(default_factory=dict)


SKILLS: dict[str, Skill] = {
    "explore_area": Skill(
        "explore_area",
        "Explore locally to reveal nearby resources and improve the shared world map.",
        ("explore", "bootstrap", "diamond"),
        {"distance": 10},
    ),
    "forage_wood": Skill(
        "forage_wood",
        "Find and mine nearby tree logs.",
        ("bootstrap", "wood", "tools"),
        {"blocks": list(WOOD_SOURCE_BLOCKS), "count": 4},
    ),
    "craft_planks": Skill(
        "craft_planks",
        "Convert gathered logs into planks.",
        ("wood", "tools"),
        {"item": "planks", "count": 1},
    ),
    "craft_sticks": Skill(
        "craft_sticks",
        "Craft sticks from planks.",
        ("tools",),
        {"item": "stick", "count": 1},
    ),
    "craft_crafting_table": Skill(
        "craft_crafting_table",
        "Craft a crafting table.",
        ("tools",),
        {"item": "crafting_table", "count": 1},
    ),
    "place_crafting_table": Skill(
        "place_crafting_table",
        "Place a crafting table where the group can use it.",
        ("tools",),
        {"item": "crafting_table"},
    ),
    "craft_wooden_pickaxe": Skill(
        "craft_wooden_pickaxe",
        "Craft and equip a wooden pickaxe.",
        ("tools", "stone"),
        {"item": "wooden_pickaxe", "count": 1},
    ),
    "mine_stone": Skill(
        "mine_stone",
        "Mine stone into cobblestone.",
        ("stone", "tools"),
        {"blocks": ["stone"], "count": 8},
    ),
    "craft_stone_pickaxe": Skill(
        "craft_stone_pickaxe",
        "Craft and equip a stone pickaxe.",
        ("tools", "iron"),
        {"item": "stone_pickaxe", "count": 1},
    ),
    "mine_iron": Skill(
        "mine_iron",
        "Find and mine iron ore.",
        ("iron", "diamond"),
        {"blocks": ["iron_ore", "deepslate_iron_ore"], "count": 3},
    ),
    "craft_furnace": Skill(
        "craft_furnace",
        "Craft a furnace for smelting ore.",
        ("iron",),
        {"item": "furnace", "count": 1},
    ),
    "smelt_iron": Skill(
        "smelt_iron",
        "Smelt iron ore into ingots.",
        ("iron", "diamond"),
        {"item": "iron_ingot", "count": 3},
    ),
    "craft_iron_pickaxe": Skill(
        "craft_iron_pickaxe",
        "Craft and equip an iron pickaxe.",
        ("diamond",),
        {"item": "iron_pickaxe", "count": 1},
    ),
    "mine_diamond": Skill(
        "mine_diamond",
        "Search for and mine diamond ore.",
        ("diamond",),
        {"blocks": ["diamond_ore", "deepslate_diamond_ore"], "count": 1},
    ),
    "share_supplies": Skill(
        "share_supplies",
        "Move near a teammate and share useful supplies.",
        ("culture", "diamond"),
        {"item": "oak_log", "count": 1},
    ),
    "escape_water": Skill(
        "escape_water",
        "Move from water or liquid-heavy terrain to nearby dry, standable ground.",
        ("recovery",),
        {"radius": 32},
    ),
    "find_crafting_spot": Skill(
        "find_crafting_spot",
        "Move to a dry open spot where blocks such as crafting tables can be placed reliably.",
        ("recovery",),
        {"radius": 24},
    ),
    "unstuck_reposition": Skill(
        "unstuck_reposition",
        "Break out of repeated pathing failures by stopping, backing up, jumping, and finding a nearby standable spot.",
        ("recovery",),
        {"radius": 18},
    ),
    "move_to_teammate": Skill(
        "move_to_teammate",
        "Move close to a teammate so agents can regroup before sharing supplies or stations.",
        ("recovery", "culture"),
        {"range": 3},
    ),
    "mine_coal": Skill(
        "mine_coal",
        "Mine coal ore for furnace fuel and torches.",
        ("stone", "iron"),
        {"blocks": ["coal_ore", "deepslate_coal_ore"], "count": 4},
    ),
}
