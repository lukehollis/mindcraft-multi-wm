import json
import os
import sys
import types
from pathlib import Path

from mindcraft.config import load_config
from mindcraft.telemetry import Telemetry


def test_telemetry_initializes_weave_and_wandb_from_env(monkeypatch, tmp_path: Path) -> None:
    weave_projects: list[str] = []
    wandb_inits: list[dict] = []
    wandb_logs: list[tuple[dict, int | None]] = []
    wandb_finished = []

    def weave_init(project: str) -> None:
        weave_projects.append(project)

    def weave_op(fn=None, *, name=None):
        def decorator(inner):
            return inner

        return decorator(fn) if fn is not None else decorator

    def wandb_init(**kwargs):
        wandb_inits.append(kwargs)
        return types.SimpleNamespace(id="run-123")

    def wandb_log(payload, step=None):
        wandb_logs.append((payload, step))

    def wandb_finish():
        wandb_finished.append(True)

    fake_weave = types.SimpleNamespace(init=weave_init, op=weave_op)
    fake_wandb = types.SimpleNamespace(init=wandb_init, log=wandb_log, finish=wandb_finish)
    monkeypatch.setitem(sys.modules, "weave", fake_weave)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WEAVE_PROJECT", "entity/weave-project")
    monkeypatch.setenv("WANDB_PROJECT", "wandb-project")
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_RUN_NAME", "mesh-telemetry-test")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_GROUP", "mesh-group")
    monkeypatch.setenv("WANDB_JOB_TYPE", "combined-replay-trainer")
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")

    cfg = load_config(
        "configs/default.yaml",
        {
            "run": {"storage_dir": tmp_path},
            "telemetry": {"enabled": True},
        },
    )

    telemetry = Telemetry(cfg)
    with telemetry.trace("unit_test", {"ok": True}):
        pass
    telemetry.log({"world_model": {"loss": 1.25}, "replay": {"size": 32}}, step=7)
    telemetry.finish()

    assert weave_projects == ["entity/weave-project"]
    assert telemetry.weave is fake_weave
    assert wandb_inits[0]["project"] == "wandb-project"
    assert wandb_inits[0]["entity"] == "entity"
    assert wandb_inits[0]["name"] == "mesh-telemetry-test"
    assert wandb_inits[0]["mode"] == "offline"
    assert wandb_inits[0]["dir"] == str(tmp_path)
    assert wandb_inits[0]["group"] == "mesh-group"
    assert wandb_inits[0]["job_type"] == "combined-replay-trainer"
    assert wandb_logs == [({"world_model/loss": 1.25, "replay/size": 32}, 7)]
    assert wandb_finished == [True]

    status_lines = (tmp_path / "telemetry_status.jsonl").read_text(encoding="utf-8").splitlines()
    statuses = [json.loads(line) for line in status_lines]
    assert statuses[0]["event"] == "init"
    assert statuses[0]["weave"]["initialized"] is True
    assert statuses[0]["wandb"]["initialized"] is True
    assert statuses[0]["wandb"]["api_key_present"] is True
    assert statuses[-1]["event"] == "finish"


def test_telemetry_removes_empty_wandb_entity_before_weave(monkeypatch, tmp_path: Path) -> None:
    def weave_init(project: str) -> None:
        assert project == "mindcraft"
        assert "WANDB_ENTITY" not in os.environ

    def wandb_init(**kwargs):
        assert kwargs["entity"] is None
        return types.SimpleNamespace(id="run-empty-entity")

    fake_weave = types.SimpleNamespace(init=weave_init, op=lambda fn=None, **_: fn or (lambda inner: inner))
    fake_wandb = types.SimpleNamespace(init=wandb_init, log=lambda *_, **__: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "weave", fake_weave)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_ENTITY", "")
    monkeypatch.setenv("WEAVE_PROJECT", "mindcraft")
    monkeypatch.setenv("WANDB_PROJECT", "mindcraft")
    monkeypatch.setenv("WANDB_MODE", "offline")

    cfg = load_config("configs/default.yaml", {"run": {"storage_dir": tmp_path}, "telemetry": {"enabled": True}})

    telemetry = Telemetry(cfg)

    assert telemetry.weave is fake_weave
    assert telemetry.wandb is fake_wandb
