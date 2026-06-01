

https://github.com/user-attachments/assets/1a893b56-da7c-4ae2-8af9-9d4bd36efe4d


# Mindcraft: Multi-agent World Model for Continuous Learning Agents

<p align="left">
  <a href="https://github.com/lukehollis/mindcraft-multi-wm">
    <img alt="GitHub repository" src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="http://mindcraft.fyi/">
    <img alt="Live demo" src="https://img.shields.io/badge/Live%20Demo-mindcraft.fyi-22c55e?style=for-the-badge&logo=vercel&logoColor=white">
  </a>
</p>

Continuous learning agents inside Minecraft based on [Andrej Karpathy’s Animals v. Ghosts](https://karpathy.bearblog.dev/animals-vs-ghosts/).

In Andrej Karpathy's blog "Animals v. Ghosts," he argues how LLMs are more like ghosts than animals because their intelligence is distilled from existing human documents. He writes, "We do not in fact have an actual, single, clean, actually bitter lesson pilled, 'turn the crank' algorithm that you could unleash upon the world and see it learn automatically from experience alone." 

Mindcraft is a runnable multi-agent Minecraft orchestration harness where generic agents learn from their own embodied experience. Each agent chooses skills, scores outcomes, stores replay, updates persistent skill memory, and trains an action-conditioned world model across episodes. Agents improve by imagining their own Minecraft rollouts within their continual-learning harness. 

W&B tracks rewards, success rates, replay growth, inventory progress, skill values, and world-model losses. Weave traces agent steps and orchestration calls for multi-agent behavior.


https://github.com/user-attachments/assets/46ebdd74-aa1f-4a48-922b-768d0dd6f267


The model is a Mamba-style SSM world model with FSQ latent codes, JEPA-style latent prediction, and MuZero-style reward/value/policy heads. 

In the future this type of continual learning will be used in robotic systems such as Unitree Go2 Pro to learn and generalize to new tasks and environments.


## Model Architecture

<img width="1920" height="1220" alt="continual_learning_world_model_architecture" src="https://github.com/user-attachments/assets/6a764500-ae89-4b54-835d-163bff702629" />


## Realworld Use 

In the realworld, it will be important for robots like the Unitree Go2 Pro to have continous learning from their environment with a method like our small Mindcraft model. We used Isaac Sim and the Go2 Pro robot to implement our continuous learning model in a simulated environment.



https://github.com/user-attachments/assets/2f3b216f-e69b-4d0a-972c-71141a420b4b




This is our real continual learning world model running on the Go2 in Isaac Lab.

Many robotic systems will need to coordinate to achieve tasks in the same way our Minecraft characters coordinate to trade, mine, craft, etc. That's why our multi-agent orchestration agent harness is important for real-world task completion.


## Code

- `mindcraft.world_model`: selective-Mamba 2 based SSM dynamics model with FSQ latent codes, JEPA-style latent prediction, MuZero-style reward/value/policy heads, and optional LoRA adapters.
- `mindcraft.replay`: persistent JSONL replay buffer with per-agent sequence windows, validation holdout windows, and progression-aware sampling.
- `mindcraft.skill_library`: learned skill values, preconditions, curiosity scores, and curriculum candidates.
- `mindcraft.planning`: MCTS planner that rolls candidate skills through the world model and penalizes uncertain predictions.
- `mindcraft.config`: YAML-backed run, learning, and telemetry config dataclasses.
- `mindcraft.robotics`: Isaac Lab Unitree Go2 continual-learning replay, skill library, and world-model trainer.
- `mindcraft.telemetry`: optional W&B/Weave status, traces, and metric logging.
- `mindcraft.training_logs`: JSONL metrics and optional TensorBoard logging.
- `scripts/replay_multiplexer.py`: merge multiple replay shards into one deduplicated training buffer.
- `scripts/isaac_go2_continual_demo.py`: Isaac Lab demo that trains the robotics world model online.
- `dashboard/`: live Next.js dashboard for agent feeds, world camera, society map, activity, and learning/progress graphs.
- `figures/`: local presentation and architecture assets.


## W&B 

We use W&B for tracking the continual learning while observing the environment. In production, this could easily be adapted to robotic systems for a diverse range of objectives.

<img width="3004" height="2002" alt="wandb_logging" src="https://github.com/user-attachments/assets/22cd6da0-c3f4-448e-8b6f-447f9c45499f" />



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
mindcraft train-replay --storage-dir runs/default --telemetry --telemetry-run-name mindcraft-replay
mindcraft train-replay --storage-dir runs/default --follow --batches 0
mindcraft train-replay --storage-dir runs/default --no-hindsight-relabeling --no-frontier-sampling
mindcraft telemetry-check --storage-dir runs/default --run-name telemetry-check
mindcraft device-info
```

The main artifacts are:

- `world_model.pt`
- `world_model_checkpoint.json`
- `training_metrics.jsonl`
- `tensorboard/` when `--tensorboard` is set

## Robotics Demo

The Isaac Lab demo mirrors the Minecraft continual-learning loop for Unitree Go2 velocity-control skills.

```sh
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --headless \
  --num_envs 4
```

Use `--video` to record an mp4 rollout, and `--cooperative-goal` to coordinate vectorized Go2 agents toward a shared formation target. Local demo outputs were migrated under
`runs/isaac_go2_video`, `runs/isaac_go2_video_policy`,
`runs/isaac_go2_video_policy_10s`, and `runs/isaac_go2_multiagent_10s`.
The latest demo also writes `go2_multiagent_orchestration.jsonl` for shared
multi-agent skill selection traces.

See `docs/isaac_go2_continual_demo.md` for the full run options and outputs.

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
