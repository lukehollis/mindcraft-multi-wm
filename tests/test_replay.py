from pathlib import Path
import json
import random

from mindcraft.replay import ReplayBuffer
from mindcraft.schemas import Observation, SkillResult, Transition


def make_transition(agent: str, step: int, episode: int = 0, skill: str = "forage_wood", success: bool = True) -> Transition:
    before = Observation(
        bot=agent,
        position=(float(step), 64.0, 0.0),
        inventory={"oak_log": step},
        nearby_blocks={"oak_log": 3},
    )
    after = Observation(
        bot=agent,
        position=(float(step + 1), 64.0, 0.0),
        inventory={"oak_log": step + 1},
        nearby_blocks={"oak_log": 2},
    )
    return Transition(
        episode=episode,
        step=step,
        agent=agent,
        role="agent",
        goal="bootstrap",
        skill=skill,
        observation=before,
        result=SkillResult(skill=skill, success=success),
        reward=1.0 if success else -0.75,
        next_observation=after,
    )


def test_replay_samples_single_agent_trajectories(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path / "experience.jsonl", capacity=100)
    for step in range(5):
        replay.append(make_transition("agent_0", step))
        replay.append(make_transition("agent_1", step))

    sequences = replay.sample_sequences(batch_size=12, sequence_length=3, rng=random.Random(3))

    assert len(sequences) == 12
    for seq in sequences:
        assert len({transition.episode for transition in seq}) == 1
        assert len({transition.agent for transition in seq}) == 1
        assert [transition.step for transition in seq] == list(range(seq[0].step, seq[0].step + 3))


def test_replay_requires_enough_steps_for_one_agent(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path / "experience.jsonl", capacity=100)
    for step in range(4):
        replay.append(make_transition("agent_0", step))
        replay.append(make_transition("agent_1", step))

    assert not replay.can_sample_sequence(5)
    assert replay.sample_sequences(batch_size=4, sequence_length=5, rng=random.Random(1)) == []


def test_replay_stratifies_across_agents_skills_and_outcomes(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path / "experience.jsonl", capacity=100)
    for step in range(6):
        replay.append(make_transition("agent_0", step, skill="forage_wood", success=step % 2 == 0))
        replay.append(make_transition("agent_1", step, skill="craft_planks", success=True))
        replay.append(make_transition("agent_1", step + 20, skill="craft_wooden_pickaxe", success=False))

    sequences = replay.sample_sequences(batch_size=9, sequence_length=3, rng=random.Random(5))
    buckets = {(seq[-1].agent, seq[-1].skill, seq[-1].result.success) for seq in sequences}

    assert ("agent_0", "forage_wood", True) in buckets
    assert ("agent_0", "forage_wood", False) in buckets
    assert ("agent_1", "craft_planks", True) in buckets
    assert ("agent_1", "craft_wooden_pickaxe", False) in buckets


def test_replay_prioritizes_progression_milestones(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path / "experience.jsonl", capacity=100)
    for step in range(8):
        replay.append(make_transition("agent_0", step, skill="explore_area", success=True))
    before = Observation(
        bot="agent_0",
        position=(8.0, 64.0, 0.0),
        inventory={"oak_planks": 3, "stick": 2, "crafting_table": 1},
    )
    after = Observation(
        bot="agent_0",
        position=(9.0, 64.0, 0.0),
        inventory={"wooden_pickaxe": 1},
    )
    replay.append(
        Transition(
            episode=0,
            step=8,
            agent="agent_0",
            role="agent",
            goal="tools",
            skill="craft_wooden_pickaxe",
            observation=before,
            result=SkillResult(skill="craft_wooden_pickaxe", success=True),
            reward=12.0,
            next_observation=after,
        )
    )

    sequences = replay.sample_sequences(batch_size=3, sequence_length=2, rng=random.Random(9))

    assert any(seq[-1].skill == "craft_wooden_pickaxe" for seq in sequences)


def test_replay_exposes_validation_holdout(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path / "experience.jsonl", capacity=100)
    for step in range(8):
        replay.append(make_transition("agent_0", step))

    sequences = replay.sample_validation_sequences(batch_size=2, sequence_length=3, rng=random.Random(2))

    assert len(sequences) == 2
    assert all(len(seq) == 3 for seq in sequences)


def test_replay_refresh_loads_appended_transitions(tmp_path: Path) -> None:
    path = tmp_path / "experience.jsonl"
    writer = ReplayBuffer(path, capacity=100)
    reader = ReplayBuffer(path, capacity=100)

    writer.append(make_transition("agent_0", 0))
    writer.append(make_transition("agent_0", 1))

    assert len(reader) == 0
    assert reader.refresh() == 2
    assert len(reader) == 2
    assert reader.refresh() == 0


def test_replay_refresh_waits_for_partial_json_line(tmp_path: Path) -> None:
    path = tmp_path / "experience.jsonl"
    writer = ReplayBuffer(path, capacity=100)
    writer.append(make_transition("agent_0", 0))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"episode":')

    reader = ReplayBuffer(path, capacity=100)

    assert len(reader) == 1
    assert reader.refresh() == 0


def test_replay_normalizes_empty_combat_kill_rewards(tmp_path: Path) -> None:
    path = tmp_path / "experience.jsonl"
    transition = make_transition("agent_0", 0, skill="duel_teammate", success=True)
    transition.result = SkillResult(
        skill="duel_teammate",
        success=True,
        data={"killed": True, "target": "agent_1", "transferred": {}, "kill_reward": 35},
    )
    transition.reward = 35.0
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(transition.to_jsonable()) + "\n")

    replay = ReplayBuffer(path, capacity=100)

    loaded = replay.tail(1)[0]
    assert loaded.reward == 2.0
    assert loaded.result.data["empty_kill"]
    assert loaded.result.data["kill_reward"] == 2.0
