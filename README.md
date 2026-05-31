# Mindcraft Multi-WM

Mindcraft is the lean world-model slice from the earlier Minecraft agent harness. It keeps the real learning path: JSONL replay, skill affordances, a PyTorch action-conditioned dynamics model, uncertainty-aware reward/value heads, checkpointing, and model-based MCTS over Minecraft skills.

## What Is Included

- `mindcraft.world_model`: selective-SSM dynamics model with FSQ latent codes, JEPA-style latent prediction, MuZero-style reward/value/policy heads, and optional LoRA adapters.
- `mindcraft.replay`: persistent JSONL replay buffer with per-agent sequence windows, validation holdout windows, and progression-aware sampling.
- `mindcraft.skill_library`: learned skill values, preconditions, curiosity scores, and curriculum candidates.
- `mindcraft.planning`: MCTS planner that rolls candidate skills through the world model and penalizes uncertain predictions.
- `mindcraft.training_logs`: JSONL metrics and optional TensorBoard logging.
- `dashboard/`: live Next.js dashboard for agent feeds, world camera, society map, activity, and learning/progress graphs.

The live bridge, dashboard, and old orchestration code are intentionally not part of this repo.

## Setup

```sh
python3 -m pip install -e '.[dev]'
pytest
```

## Train From Replay

Replay files are newline-delimited `Transition.to_jsonable()` payloads. By default the CLI reads `<storage-dir>/experience.jsonl` and writes checkpoints plus metrics back to `<storage-dir>`.

```sh
mindcraft train-replay \
  --storage-dir runs/default \
  --batches 500 \
  --batch-size 16 \
  --sequence-length 8
```

Useful flags:

```sh
mindcraft train-replay --storage-dir runs/default --device cuda --tensorboard
mindcraft train-replay --storage-dir runs/default --follow --batches 0
mindcraft device-info
```

The main artifacts are:

- `world_model.pt`
- `world_model_checkpoint.json`
- `training_metrics.jsonl`
- `tensorboard/` when `--tensorboard` is set

## Dashboard

The dashboard is a standalone Next app that consumes the same live snapshot API as the original harness:

- `GET /snapshot`
- `WS /stream`

Run it from the dashboard directory:

```sh
cd dashboard
npm install
npm run dev
```

By default it serves on `http://localhost:8790` and reads telemetry from the same host. Override the data source when the bridge or snapshot server runs elsewhere:

```sh
NEXT_PUBLIC_BRIDGE_HTTP=http://localhost:8780 \
NEXT_PUBLIC_BRIDGE_WS=ws://localhost:8780/stream \
npm run dev
```
