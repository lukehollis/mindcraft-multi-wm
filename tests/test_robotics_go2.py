from pathlib import Path
import random

from mindcraft.robotics import (
    ROBOTICS_SKILLS,
    RoboticsReplayBuffer,
    RoboticsSkillLibrary,
    RoboticsTransition,
    RoboticsWorldModelTrainer,
    choose_robotics_skill,
    encode_robotics_action,
)


def make_transition(step: int, skill: str = "walk_forward") -> RoboticsTransition:
    obs = [step / 10.0, 0.1, 0.2, 0.3, 0.4]
    next_obs = [(step + 1) / 10.0, 0.1, 0.2, 0.3, 0.4]
    return RoboticsTransition(
        episode=0,
        step=step,
        env_id=0,
        skill=skill,
        command=ROBOTICS_SKILLS[skill].command,
        observation=obs,
        reward=0.75,
        next_observation=next_obs,
        done=False,
        info={"progress_m": 0.1},
    )


def test_robotics_action_encoder() -> None:
    encoded = encode_robotics_action("walk_forward")

    assert encoded.sum() == 1.0
    assert encoded.ndim == 1


def test_robotics_replay_samples_sequences(tmp_path: Path) -> None:
    replay = RoboticsReplayBuffer(tmp_path / "go2_experience.jsonl")
    for step in range(10):
        replay.append(make_transition(step))

    sampled = replay.sample_sequences(batch_size=2, sequence_length=4, rng=random.Random(0))

    assert len(sampled) == 2
    assert all(len(seq) == 4 for seq in sampled)


def test_robotics_world_model_trains_one_batch(tmp_path: Path) -> None:
    seq = [make_transition(step) for step in range(4)]
    trainer = RoboticsWorldModelTrainer(tmp_path, obs_dim=5, d_model=32, layers=1, heads=4, device="cpu")

    metrics = trainer.train_batches([seq])

    assert metrics is not None
    assert metrics.loss > 0
    assert metrics.train_step == 1
    prediction = trainer.predict_skill(seq[-1].next_observation, "walk_forward")
    assert prediction["next_obs"].shape == (5,)
    trainer.save()
    assert (tmp_path / "go2_world_model.pt").exists()


def test_robotics_skill_library_and_selector(tmp_path: Path) -> None:
    library = RoboticsSkillLibrary(tmp_path / "go2_skill_library.json")
    library.update("walk_forward", reward=1.0, success=True, episode=0, step=0, prediction_error=0.2)
    library.save()

    loaded = RoboticsSkillLibrary(tmp_path / "go2_skill_library.json")
    skill, planner, diagnostics = choose_robotics_skill([0.0] * 5, loaded, random.Random(1), epsilon=0.0)

    assert skill in ROBOTICS_SKILLS
    assert planner == "skill_library"
    assert diagnostics == {}
