from pathlib import Path
import random

from mindcraft.skill_library import SkillLibrary
from mindcraft.schemas import Observation


def test_skill_library_respects_crafting_preconditions(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(
        bot="agent_0",
        inventory={"oak_log": 2},
        nearby_blocks={"oak_log": 4},
    )

    candidates = lib.candidates("agent_0", "wood", obs)

    assert "craft_planks" in candidates
    assert "craft_wooden_pickaxe" not in candidates


def test_skill_library_updates_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    lib = SkillLibrary(path)
    lib.update("forage_wood", reward=3.0, success=True, episode=0, step=1, prediction_error=0.5, agent_id="agent_0")
    lib.save()

    loaded = SkillLibrary(path)

    assert loaded.stats["forage_wood"].attempts == 1
    assert loaded.stats["forage_wood"].successes == 1
    assert loaded.stats["forage_wood"].q_value > 0
    assert loaded.agent_stats["agent_0"]["forage_wood"].attempts == 1


def test_skill_library_migrates_old_explore_skill_name(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        '{"scout_area":{"attempts":1,"successes":1,"total_reward":1.0,"q_value":1.0,'
        '"prediction_error":0.5,"last_episode":0,"last_step":0,"notes":[]}}',
        encoding="utf-8",
    )

    loaded = SkillLibrary(path)

    assert loaded.stats["explore_area"].attempts == 1


def test_skill_selection_uses_affordance_bonus(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={}, nearby_blocks={"oak_log": 12})

    choice = lib.select("agent_0", "bootstrap", obs, random.Random(1), epsilon=0.0, ucb_c=0.0, curiosity_weight=0.0)

    assert choice == "forage_wood"


def test_skill_library_exposes_deterministic_scores(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={"oak_log": 2}, nearby_blocks={"oak_log": 12})

    scores = lib.score_candidates("agent_0", "wood", obs, ucb_c=0.0, curiosity_weight=0.0)
    bonuses = lib.affordance_bonuses("agent_0", "wood", obs)

    assert scores["craft_planks"] >= 5.0
    assert max(scores, key=scores.get) == "craft_planks"
    assert bonuses["craft_planks"] >= 5.0


def test_skill_library_handles_non_oak_wood_families(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(
        bot="agent_0",
        inventory={"dark_oak_planks": 4},
        nearby_blocks={"dark_oak_log": 12},
    )

    candidates = lib.candidates("agent_0", "tools", obs)

    assert "craft_sticks" in candidates
    assert "craft_crafting_table" in candidates


def test_empty_bootstrap_does_not_share_missing_supplies(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={}, nearby_blocks={"dark_oak_log": 12})

    candidates = lib.candidates("agent_0", "bootstrap", obs)
    choice = lib.select("agent_0", "bootstrap", obs, random.Random(1), epsilon=0.0, ucb_c=0.0, curiosity_weight=0.0)

    assert "share_supplies" not in candidates
    assert choice == "forage_wood"


def test_share_supplies_is_not_basic_tool_bootstrap(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={"dark_oak_log": 8}, nearby_blocks={"dark_oak_log": 12})

    assert "share_supplies" not in lib.candidates("agent_0", "wood", obs)
    assert "share_supplies" not in lib.candidates("agent_0", "tools", obs)


def test_share_supplies_requires_surplus_inventory(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    low_value = Observation(
        bot="agent_0",
        inventory={"stick": 2, "wooden_pickaxe": 1, "cobblestone": 13, "dark_oak_log": 1},
        nearby_blocks={"stone": 12},
    )
    surplus = Observation(
        bot="agent_1",
        inventory={"oak_log": 24},
        nearby_blocks={"stone": 12},
    )

    assert "share_supplies" not in lib.candidates("agent_0", "iron", low_value)
    assert "share_supplies" in lib.candidates("agent_1", "diamond", surplus)


def test_skill_preferences_are_learned_per_agent(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={}, nearby_blocks={"stone": 4})
    lib.update("explore_area", reward=8.0, success=True, episode=0, step=1, agent_id="agent_0")
    lib.update("forage_wood", reward=8.0, success=True, episode=0, step=1, agent_id="agent_1")

    agent_0_scores = lib.score_candidates("agent_0", "bootstrap", obs, ucb_c=0.0, curiosity_weight=0.0)
    agent_1_scores = lib.score_candidates("agent_1", "bootstrap", obs, ucb_c=0.0, curiosity_weight=0.0)

    assert agent_0_scores["explore_area"] > agent_0_scores["forage_wood"]
    assert agent_1_scores["forage_wood"] > agent_1_scores["explore_area"]


def test_iron_goal_does_not_backslide_to_wood_gathering(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(
        bot="agent_0",
        inventory={"stone_pickaxe": 1, "crafting_table": 1, "furnace": 1, "oak_log": 64},
        nearby_blocks={"stone": 12},
    )

    candidates = lib.candidates("agent_0", "iron", obs)

    assert "mine_iron" in candidates
    assert "forage_wood" not in candidates


def test_furnace_requires_crafting_table_access(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    without_table = Observation(bot="agent_0", inventory={"cobblestone": 8}, nearby_blocks={"stone": 12})
    with_table = Observation(
        bot="agent_0",
        inventory={"cobblestone": 8},
        nearby_blocks={"stone": 12, "crafting_table": 1},
    )

    assert "craft_furnace" not in lib.candidates("agent_0", "iron", without_table)
    assert "craft_furnace" in lib.candidates("agent_0", "iron", with_table)


def test_curriculum_uses_shared_cobblestone_for_stone_pickaxe(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(
        bot="agent_0",
        inventory={"cobblestone": 3, "stick": 2},
        nearby_blocks={"crafting_table": 1},
    )

    assert lib.curriculum_candidates("agent_0", "iron", obs)[0] == "craft_stone_pickaxe"


def test_recovery_skills_are_not_normal_bootstrap_candidates(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={}, nearby_blocks={"oak_log": 8})

    candidates = lib.candidates("agent_0", "bootstrap", obs)

    assert "escape_water" not in candidates
    assert "unstuck_reposition" not in candidates
    assert "forage_wood" in candidates


def test_escape_water_is_available_for_recovery_goal(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path / "skills.json")
    obs = Observation(bot="agent_0", inventory={}, nearby_blocks={"water": 96})

    candidates = lib.candidates("agent_0", "recovery", obs)
    scores = lib.score_candidates("agent_0", "recovery", obs, ucb_c=0.0, curiosity_weight=0.0)

    assert "escape_water" in candidates
    assert scores["escape_water"] >= 10.0
