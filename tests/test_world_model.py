from pathlib import Path
import json
import random

import torch

from mindcraft.schemas import Observation, SkillResult, Transition
from mindcraft.skill_library import SkillLibrary
from mindcraft.world_model import WorldModelTrainer, encode_action, encode_affordances, encode_observation, encode_unlocks
from mindcraft.planning import ModelMCTSPlanner


def make_transition(step: int) -> Transition:
    before = Observation(
        bot="agent_0",
        position=(float(step), 64.0, 0.0),
        inventory={"oak_log": step},
        nearby_blocks={"oak_log": 5, "stone": 2},
    )
    after = Observation(
        bot="agent_0",
        position=(float(step + 1), 64.0, 0.0),
        inventory={"oak_log": step + 1},
        nearby_blocks={"oak_log": 4, "stone": 2},
    )
    return Transition(
        episode=0,
        step=step,
        agent="agent_0",
        role="agent",
        goal="bootstrap",
        skill="forage_wood",
        observation=before,
        result=SkillResult(skill="forage_wood", success=True),
        reward=1.25,
        next_observation=after,
    )


def test_feature_encoders_have_expected_shapes() -> None:
    obs = Observation(bot="agent_0", inventory={"oak_log": 1})

    assert encode_observation(obs).ndim == 1
    assert encode_action("forage_wood").sum() == 1.0
    assert encode_action("mine_coal").sum() == 1.0
    assert encode_unlocks(obs).ndim == 1
    assert encode_affordances(obs).ndim == 1


def test_world_model_trains_one_batch(tmp_path: Path) -> None:
    trainer = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    seq = [make_transition(i) for i in range(4)]

    metrics = trainer.train_batches([seq])

    assert metrics is not None
    assert metrics.loss > 0
    assert metrics.train_step == 1
    trainer.save()
    assert (tmp_path / "world_model.pt").exists()
    assert (tmp_path / "world_model_checkpoint.json").exists()


def test_world_model_predicts_skill_for_planning(tmp_path: Path) -> None:
    trainer = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    seq = [make_transition(i) for i in range(4)]
    trainer.train_batches([seq])

    prediction = trainer.predict_skill(seq[-1].next_observation, "forage_wood")

    assert {
        "next_obs",
        "reward",
        "reward_uncertainty",
        "value",
        "value_uncertainty",
        "model_uncertainty",
        "done_logit",
        "prior",
        "unlock",
        "affordance",
        "skill_affordance",
        "unlock_delta",
        "unlock_gain",
    }.issubset(prediction)
    assert prediction["next_obs"].shape == encode_observation(seq[-1].next_observation).shape
    assert prediction["model_uncertainty"] >= 0.0
    assert "logs" in prediction["unlock"]
    assert "mine_coal" in prediction["affordance"]


def test_world_model_checkpoint_resumes_training_step(tmp_path: Path) -> None:
    trainer = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    seq = [make_transition(i) for i in range(4)]

    first = trainer.train_batches([seq])
    assert first is not None
    trainer.save()

    payload = torch.load(tmp_path / "world_model.pt", map_location="cpu")
    metadata = json.loads((tmp_path / "world_model_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["train_step"] == 1
    assert payload["checkpoint_version"] == 3
    assert metadata["train_step"] == 1
    assert metadata["checkpoint_version"] == 3

    resumed = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    assert resumed.train_step == 1
    assert resumed.last_loss == first.loss

    second = resumed.train_batches([seq])
    assert second is not None
    assert second.train_step == 2


def test_world_model_reloads_external_checkpoint_when_changed(tmp_path: Path) -> None:
    source = WorldModelTrainer(
        tmp_path,
        d_model=32,
        layers=1,
        heads=4,
        lr=1e-3,
        device="cpu",
        checkpoint_name="central.pt",
    )
    seq = [make_transition(i) for i in range(4)]
    trained = source.train_batches([seq])
    assert trained is not None
    source.save()

    actor = WorldModelTrainer(
        tmp_path,
        d_model=32,
        layers=1,
        heads=4,
        lr=1e-3,
        device="cpu",
        checkpoint_name="actor.pt",
    )

    assert actor.reload_if_changed(tmp_path / "central.pt")
    assert actor.train_step == source.train_step
    assert not actor.reload_if_changed(tmp_path / "central.pt")


def test_world_model_reports_validation_metrics(tmp_path: Path) -> None:
    trainer = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    seq = [make_transition(i) for i in range(4)]
    val_seq = [make_transition(i + 4) for i in range(4)]

    metrics = trainer.train_batches([seq], validation_sequences=[val_seq])

    assert metrics is not None
    assert metrics.val_loss is not None
    assert metrics.val_reward_loss is not None
    assert metrics.val_unlock_loss is not None
    assert metrics.val_affordance_loss is not None


def test_mcts_planner_returns_candidate_skill(tmp_path: Path) -> None:
    trainer = WorldModelTrainer(tmp_path, d_model=32, layers=1, heads=4, lr=1e-3, device="cpu")
    seq = [make_transition(i) for i in range(4)]
    trainer.train_batches([seq])
    library = SkillLibrary(tmp_path / "skills.json")
    planner = ModelMCTSPlanner(trainer, simulations=4, depth=2)

    plan = planner.select(seq[-1].next_observation, ["forage_wood", "explore_area"], library, rng=random.Random(0))

    assert plan.skill in {"forage_wood", "explore_area"}
    assert plan.visits > 0
    assert "uncertainty" in plan.diagnostics[plan.skill]
