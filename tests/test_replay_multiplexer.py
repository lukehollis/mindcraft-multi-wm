from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mindcraft.schemas import Observation, SkillResult, Transition


def make_transition(agent: str, step: int) -> Transition:
    before = Observation(bot=agent, inventory={"oak_log": step})
    after = Observation(bot=agent, inventory={"oak_log": step + 1})
    return Transition(
        episode=0,
        step=step,
        agent=agent,
        role="agent",
        goal="bootstrap",
        skill="forage_wood",
        observation=before,
        result=SkillResult(skill="forage_wood", success=True),
        reward=1.0,
        next_observation=after,
    )


def test_replay_multiplexer_deduplicates_sources(tmp_path: Path) -> None:
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    output = tmp_path / "combined" / "experience.jsonl"
    rows = [
        make_transition("agent_0", 0).to_jsonable(),
        make_transition("agent_0", 1).to_jsonable(),
    ]
    source_a.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    source_b.write_text(json.dumps(rows[1], sort_keys=True) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/replay_multiplexer.py",
            "--output",
            str(output),
            "--source",
            str(source_a),
            "--source",
            str(source_b),
        ],
        check=True,
    )

    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(merged) == 2
    assert [row["step"] for row in merged] == [0, 1]


def test_replay_multiplexer_ignores_partial_source_line(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "combined" / "experience.jsonl"
    source.write_text(
        json.dumps(make_transition("agent_0", 0).to_jsonable(), sort_keys=True) + "\n{\"episode\":",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/replay_multiplexer.py",
            "--output",
            str(output),
            "--source",
            str(source),
        ],
        check=True,
    )

    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(merged) == 1
    assert merged[0]["step"] == 0
