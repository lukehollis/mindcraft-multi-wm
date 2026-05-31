from __future__ import annotations

import contextlib
import json
import os
import socket
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from mindcraft.config import MindcraftConfig

F = TypeVar("F", bound=Callable[..., Any])


class Telemetry:
    def __init__(self, cfg: MindcraftConfig):
        self.cfg = cfg
        self.weave: Any | None = None
        self.wandb: Any | None = None
        self.run: Any | None = None
        self.status: dict[str, Any] = {
            "enabled": cfg.telemetry.enabled,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "storage_dir": str(cfg.run.storage_dir),
            "wandb": {"initialized": False},
            "weave": {"initialized": False},
        }
        _unset_empty_env("WANDB_ENTITY")
        if not cfg.telemetry.enabled:
            self._record_status("init")
            return
        self._init_weave()
        self._init_wandb()
        self._record_status("init")

    def op(self, fn: F) -> F:
        if self.weave is not None:
            try:
                return self.weave.op(fn)  # type: ignore[return-value]
            except Exception as exc:
                self.status["weave"]["op_error"] = _error(exc)
                self._record_status("weave_op_error")
        return fn

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        flat = _flatten(metrics)
        if self.wandb is not None and self.run is not None:
            try:
                self.wandb.log(flat, step=step)
            except Exception as exc:
                self.status["wandb"]["log_error"] = _error(exc)
                self._record_status("wandb_log_error")

    @contextlib.contextmanager
    def trace(self, name: str, payload: dict[str, Any] | None = None) -> Iterator[None]:
        if self.weave is None:
            yield
            return
        op_name = f"trace_{name}"

        @self.weave.op(name=op_name)
        def _trace_op(data: dict[str, Any]) -> dict[str, Any]:
            return data

        try:
            _trace_op(payload or {})
        except Exception as exc:
            self.status["weave"]["trace_error"] = _error(exc)
            self._record_status("weave_trace_error")
        yield

    def finish(self) -> None:
        if self.wandb is not None and self.run is not None:
            try:
                self.wandb.finish()
            except Exception as exc:
                self.status["wandb"]["finish_error"] = _error(exc)
        self._record_status("finish")

    def _init_weave(self) -> None:
        if os.getenv("WANDB_MODE", self.cfg.telemetry.wandb_mode) == "offline" and not _env("WEAVE_PROJECT"):
            self.status["weave"] = {"initialized": False, "reason": "offline_mode"}
            return
        try:
            import weave
        except Exception as exc:
            self.status["weave"] = {"initialized": False, "reason": "import_failed", "error": _error(exc)}
            return
        candidates = []
        env_project = _env("WEAVE_PROJECT")
        if env_project:
            candidates.append(env_project)
        entity = _env("WANDB_ENTITY") or self.cfg.telemetry.wandb_entity
        wandb_project = _env("WANDB_PROJECT") or self.cfg.telemetry.wandb_project
        if entity:
            candidates.append(f"{entity}/{wandb_project}")
        candidates.append(self.cfg.telemetry.weave_project)
        candidates.append(wandb_project)
        errors = []
        for project in dict.fromkeys(project for project in candidates if project):
            try:
                weave.init(project)
            except Exception as exc:
                errors.append({"project": project, "error": _error(exc)})
                continue
            self.weave = weave
            self.status["weave"] = {"initialized": True, "project": project}
            print(f"weave initialized: project={project}")
            return
        self.status["weave"] = {"initialized": False, "reason": "init_failed", "attempts": errors}

    def _init_wandb(self) -> None:
        try:
            import wandb
        except Exception as exc:
            self.status["wandb"] = {"initialized": False, "reason": "import_failed", "error": _error(exc)}
            return
        project = _env("WANDB_PROJECT") or self.cfg.telemetry.wandb_project
        entity = _env("WANDB_ENTITY") or self.cfg.telemetry.wandb_entity
        mode = os.getenv("WANDB_MODE", self.cfg.telemetry.wandb_mode)
        run_name = _env("WANDB_RUN_NAME") or self.cfg.telemetry.run_name
        group = _env("WANDB_GROUP")
        job_type = _env("WANDB_JOB_TYPE")
        storage_dir = Path(self.cfg.run.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.run = wandb.init(
                project=project,
                entity=entity or None,
                name=run_name,
                mode=mode,
                config=_jsonable(asdict(self.cfg)),
                dir=str(storage_dir),
                group=group,
                job_type=job_type,
            )
        except Exception as exc:
            self.run = None
            self.status["wandb"] = {
                "initialized": False,
                "reason": "init_failed",
                "project": project,
                "entity": entity,
                "mode": mode,
                "api_key_present": bool(os.getenv("WANDB_API_KEY")),
                "error": _error(exc),
            }
            return
        self.wandb = wandb
        self.status["wandb"] = {
            "initialized": True,
            "project": project,
            "entity": entity,
            "mode": mode,
            "run_name": run_name,
            "group": group,
            "job_type": job_type,
            "api_key_present": bool(os.getenv("WANDB_API_KEY")),
        }
        run_id = getattr(self.run, "id", None)
        if run_id is not None:
            self.status["wandb"]["run_id"] = run_id
        print(f"wandb initialized: project={project} mode={mode} run_name={run_name}")

    def _record_status(self, event: str) -> None:
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.status,
        }
        try:
            path = Path(self.cfg.run.storage_dir) / "telemetry_status.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        except Exception as exc:
            print(f"telemetry status write failed: {_error(exc)}")


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[path] = value
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _unset_empty_env(name: str) -> None:
    if os.environ.get(name) == "":
        os.environ.pop(name, None)
