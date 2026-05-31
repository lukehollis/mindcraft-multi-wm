

https://github.com/user-attachments/assets/1a893b56-da7c-4ae2-8af9-9d4bd36efe4d


# Mindcraft: Multi-agent World Model for Continuous Learning Agents

Continuous learning agents inside Minecraft based on Andrej Karpathy’s Animals v. Ghosts

In Andrej Karpathy's blog "Animals v. Ghosts," he argues how LLMs are more like ghosts than animals because their intelligence is distilled from existing human documents. He writes, "We do not in fact have an actual, single, clean, actually bitter lesson pilled, 'turn the crank' algorithm that you could unleash upon the world and see it learn automatically from experience alone." 

Mindcraft is a runnable multi-agent Minecraft orchestration harness where generic agents learn from their own embodied experience. Each agent chooses skills, scores outcomes, stores replay, updates persistent skill memory, and trains an action-conditioned world model across episodes. Agents improve by imagining their own Minecraft rollouts within their continual-learning harness. 

W&B tracks rewards, success rates, replay growth, inventory progress, skill values, and world-model losses. Weave traces agent steps and orchestration calls for multi-agent behavior.


https://github.com/user-attachments/assets/46ebdd74-aa1f-4a48-922b-768d0dd6f267


The model is a Mamba-style SSM world model with FSQ latent codes, JEPA-style latent prediction, and MuZero-style reward/value/policy heads. 

In the future this type of continual learning will be used in robotic systems such as Unitree G2 Pro Robot Dog to learn and generalize to new tasks and environments. 

So the real question is, do you want a continuously learning Unitree G2 Pro Robot Dog?

## What Is Included

- `mindcraft.world_model`: selective-Mamba 2 based SSM dynamics model with FSQ latent codes, JEPA-style latent prediction, MuZero-style reward/value/policy heads, and optional LoRA adapters.
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
