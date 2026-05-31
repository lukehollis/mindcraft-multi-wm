from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


Inventory = dict[str, int]


@dataclass(slots=True)
class Observation:
    bot: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 20.0
    food: float = 20.0
    inventory: Inventory = field(default_factory=dict)
    nearby_blocks: dict[str, int] = field(default_factory=dict)
    nearby_entities: dict[str, int] = field(default_factory=dict)
    equipped: str | None = None
    line_of_sight: str | None = None
    biome: str | None = None
    time_of_day: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, bot: str, payload: dict[str, Any]) -> "Observation":
        pos = payload.get("position") or {}
        if isinstance(pos, dict):
            position = (
                float(pos.get("x", 0.0)),
                float(pos.get("y", 0.0)),
                float(pos.get("z", 0.0)),
            )
        else:
            position = tuple(pos) if len(pos) == 3 else (0.0, 0.0, 0.0)
        return cls(
            bot=bot,
            position=position,
            health=float(payload.get("health", 20.0) or 0.0),
            food=float(payload.get("food", 20.0) or 0.0),
            inventory={str(k): int(v) for k, v in (payload.get("inventory") or {}).items()},
            nearby_blocks={str(k): int(v) for k, v in (payload.get("nearby_blocks") or {}).items()},
            nearby_entities={str(k): int(v) for k, v in (payload.get("nearby_entities") or {}).items()},
            equipped=payload.get("equipped"),
            line_of_sight=payload.get("line_of_sight"),
            biome=payload.get("biome"),
            time_of_day=payload.get("time_of_day"),
            raw=payload,
        )

    def count_items(self, *names: str) -> int:
        return sum(self.inventory.get(name, 0) for name in names)


@dataclass(slots=True)
class SkillResult:
    skill: str
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass(slots=True)
class Transition:
    episode: int
    step: int
    agent: str
    role: str
    goal: str
    skill: str
    observation: Observation
    result: SkillResult
    reward: float
    next_observation: Observation
    done: bool = False
    timestamp: float = field(default_factory=time)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "agent": self.agent,
            "role": self.role,
            "goal": self.goal,
            "skill": self.skill,
            "observation": _obs_to_dict(self.observation),
            "result": {
                "skill": self.result.skill,
                "success": self.result.success,
                "message": self.result.message,
                "data": self.result.data,
                "duration_s": self.result.duration_s,
            },
            "reward": self.reward,
            "next_observation": _obs_to_dict(self.next_observation),
            "done": self.done,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "Transition":
        result = data.get("result") or {}
        return cls(
            episode=int(data["episode"]),
            step=int(data["step"]),
            agent=str(data["agent"]),
            role=str(data["role"]),
            goal=str(data["goal"]),
            skill=str(data["skill"]),
            observation=_obs_from_dict(data["observation"]),
            result=SkillResult(
                skill=str(result.get("skill", data["skill"])),
                success=bool(result.get("success", False)),
                message=str(result.get("message", "")),
                data=dict(result.get("data") or {}),
                duration_s=float(result.get("duration_s", 0.0) or 0.0),
            ),
            reward=float(data["reward"]),
            next_observation=_obs_from_dict(data["next_observation"]),
            done=bool(data.get("done", False)),
            timestamp=float(data.get("timestamp", time())),
        )


def _obs_to_dict(obs: Observation) -> dict[str, Any]:
    return {
        "bot": obs.bot,
        "position": list(obs.position),
        "health": obs.health,
        "food": obs.food,
        "inventory": obs.inventory,
        "nearby_blocks": obs.nearby_blocks,
        "nearby_entities": obs.nearby_entities,
        "equipped": obs.equipped,
        "line_of_sight": obs.line_of_sight,
        "biome": obs.biome,
        "time_of_day": obs.time_of_day,
        "raw": obs.raw,
    }


def _obs_from_dict(data: dict[str, Any]) -> Observation:
    return Observation(
        bot=str(data.get("bot", "")),
        position=tuple(float(x) for x in data.get("position", [0.0, 0.0, 0.0])),
        health=float(data.get("health", 20.0) or 0.0),
        food=float(data.get("food", 20.0) or 0.0),
        inventory={str(k): int(v) for k, v in (data.get("inventory") or {}).items()},
        nearby_blocks={str(k): int(v) for k, v in (data.get("nearby_blocks") or {}).items()},
        nearby_entities={str(k): int(v) for k, v in (data.get("nearby_entities") or {}).items()},
        equipped=data.get("equipped"),
        line_of_sight=data.get("line_of_sight"),
        biome=data.get("biome"),
        time_of_day=data.get("time_of_day"),
        raw=dict(data.get("raw") or {}),
    )
