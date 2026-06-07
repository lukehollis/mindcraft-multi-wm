# Mineflayer Autoresearch Program

You are a local Codex autoresearcher running in the `animals_ghosts` repository.
Your job is to continuously improve the real Agent Lab Minecraft learning harness.

This is not a toy benchmark. Optimize actual live Mineflayer agent learning after 300 seconds of wall-clock action collection. The five-minute budget means the agents should act as fast as Mineflayer, the Minecraft server, and pathfinding can realistically act inside those 300 real seconds.

## Objective

Maximize the score produced by:

```sh
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
"$PYTHON" scripts/mineflayer_autoresearch_trial.py \
  --duration-s 300 \
  --description "<short experiment description>"
```

The trial runner:

- selects all currently ready live Mineflayer bots from `ws://localhost:8765`;
- creates a fresh `runs/autoresearch/<timestamp>-<commit>/` directory;
- runs the Python learner with huge episode and step ceilings, stopped by wall-clock time;
- evaluates replay with `scripts/agent_space_eval.py`;
- appends `runs/autoresearch/results.tsv`;
- writes `trial_summary.json` containing the deterministic behavioral score.

The score is:

- `1000 * proof_pass_count`
- `+ 100 * max(0, max_stage_delta)`
- `+ 25 * max(0, stage_advances_delta)`
- `+ 10 * max(0, avg_reward_delta)`
- `+ 5 * max(0, direct_success_rate_delta)`
- `+ 0.05 * transitions`

Tie-breakers are last max stage, all-time milestones, then transitions.

## Protected Surfaces

Do not modify these scoring surfaces during ordinary experiments:

- `scripts/mineflayer_autoresearch_trial.py`
- `scripts/agent_space_eval.py`
- `animals_ghosts/learning/evaluation.py`
- `runs/autoresearch/results.tsv`

Only modify them in a separate, explicitly named harness-change commit after you first prove the current code has a real bug. Ordinary learning improvements must come from the agent, planner, skill execution, bridge behavior, config defaults, training logic, or tests.

## Allowed Work

You may edit real implementation files, including:

- `animals_ghosts/agents/`
- `animals_ghosts/minecraft/`
- `animals_ghosts/models/`
- `animals_ghosts/planning/`
- `animals_ghosts/progression.py`
- `animals_ghosts/config.py`
- `configs/`
- `bridge/server.js`
- `docker-compose.yml`
- `scripts/` except protected scoring surfaces
- `tests/`

You may run tests, short fake-bridge preflights, live Mineflayer trials, and inspect logs/artifacts.

Do not install new dependencies unless a change is impossible without them. Prefer the existing Python, Node, Mineflayer, PyTorch, TensorBoard, W&B, and Weave stack.

## Git Safety

The working tree may contain user changes unrelated to autoresearch. Preserve them.

Rules:

- Never run `git add .`.
- Never run `git reset --hard`.
- Never run `git checkout -- <path>` unless the path is part of your own current failed experiment.
- Stage only the explicit paths you intentionally edited.
- Before each experiment, record `git status --short`.
- Commit each candidate before running the 300-second trial.
- Leave `runs/autoresearch/results.tsv` uncommitted.

If unrelated dirty files exist, ignore them.

## Experiment Loop

Loop forever until the human stops you.

1. Inspect current state:
   - `git status --short --branch`
   - read recent `runs/autoresearch/results.tsv`
   - inspect the best recent `trial_summary.json`, `agent.log`, and evaluation report.
2. Choose one real experiment likely to improve learning per minute.
3. Edit the implementation.
4. Run focused tests or a short fake preflight when useful:
   - `PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q -p no:cacheprovider`
   - or a targeted subset.
5. Stage only explicit changed implementation/test paths.
6. Commit with a concise experiment message.
7. Run the live 300-second trial through `scripts/mineflayer_autoresearch_trial.py`.
8. Read the produced `trial_summary.json`.
9. Compare against the best previous score in `runs/autoresearch/results.tsv`.
10. If the score improves, keep the commit and continue from it.
11. If the score is worse or equal, mark the result as discarded in your notes and reset only your candidate commit with `git reset --mixed HEAD~1`; keep generated run artifacts and `results.tsv`.
12. If the run crashes because of a simple bug in your change, fix it and rerun. If the idea is flawed, discard it and move on.

Do not stop after one run. Do not ask whether to continue. You are expected to keep adapting the codebase continuously.

## First Run

If `runs/autoresearch/results.tsv` has no data rows, run one baseline live trial before making any experimental code change:

```sh
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
"$PYTHON" scripts/mineflayer_autoresearch_trial.py \
  --duration-s 300 \
  --description "baseline"
```

Then begin the experiment loop.
