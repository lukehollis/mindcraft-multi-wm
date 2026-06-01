# Isaac Lab Unitree Go2 Continual-Learning Demo

This demo reuses the continual-learning idea from Mindcraft in a robotics setting:

- Isaac Lab supplies the default Unitree Go2 velocity-tracking scene.
- A low-level RSL-RL policy turns velocity commands into joint actions.
- The Mindcraft world model learns action-conditioned robot dynamics online from replay.
- A persistent skill library and model-based selector choose high-level locomotion skills.
- Multiple vectorized Go2 agents can learn together through one shared replay buffer, skill
  library, and world model.
- An optional cooperative rally task gives the team a shared target: the centroid must enter
  the goal zone while each Go2 moves into its assigned formation slot.

The hackathon prize text says "Unitree G2 Pro"; in this Isaac Lab checkout the robot dog asset and registered tasks are named `Unitree Go2`.

## Run

Run through Isaac Lab's launcher, not plain `python`, so Omniverse/Isaac packages are initialized:

```sh
cd /home/trantor/Research/mindcraft-multi-wm
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --headless \
  --num_envs 4 \
  --episodes 3 \
  --skills-per-episode 80 \
  --skill-horizon 24
```

To record a 10-second multi-agent headless mp4 with the camera framing all Go2 agents:

```sh
cd /home/trantor/Research/mindcraft-multi-wm
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --headless \
  --video \
  --video-length 500 \
  --video-dir runs/isaac_go2_multiagent_10s/videos \
  --storage-dir runs/isaac_go2_multiagent_10s \
  --num_envs 4 \
  --camera-mode group \
  --episodes 1 \
  --skills-per-episode 24 \
  --skill-horizon 24
```

To render the cooperative-goal version, add the shared rally objective:

```sh
cd /home/trantor/Research/mindcraft-multi-wm
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --headless \
  --video \
  --video-length 500 \
  --video-dir runs/isaac_go2_cooperative_goal_10s/videos \
  --storage-dir runs/isaac_go2_cooperative_goal_10s \
  --num_envs 4 \
  --camera-mode group \
  --cooperative-goal \
  --goal-offset-x 0.65 \
  --goal-radius 0.35 \
  --episodes 1 \
  --skills-per-episode 24 \
  --skill-horizon 24
```

By default the script tries Isaac Lab's published RSL-RL checkpoint for:

```text
Isaac-Velocity-Flat-Unitree-Go2-Play-v0
```

If you already trained a controller, point to it directly:

```sh
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --policy-checkpoint /path/to/checkpoint.pt
```

For a dependency/plumbing check only, you can allow zero actions:

```sh
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --no-pretrained-policy \
  --allow-zero-policy \
  --headless \
  --episodes 1 \
  --skills-per-episode 4
```

## Outputs

The default output directory is `runs/isaac_go2`:

- `go2_experience.jsonl`: robot transitions from Isaac Lab.
- `go2_skill_library.json`: persistent skill values, success rates, and prediction errors.
- `go2_world_model.pt`: action-conditioned world-model checkpoint.
- `go2_world_model_checkpoint.json`: checkpoint metadata.
- `go2_training_metrics.jsonl`: online world-model losses.
- `go2_multiagent_orchestration.jsonl`: per-window multi-agent selections, rewards,
  progress, planner counts, and shared replay/model state.
- `go2_cooperative_goal_summary.json`: final shared-goal state and first achieved step
  when `--cooperative-goal` is enabled.
- `videos/*.mp4`: optional Isaac Lab renderings when `--video` is set.

## What Is Being Learned

The high-level action space is a discrete robotics skill vocabulary:

```text
stand, walk_forward_slow, walk_forward, walk_backward,
strafe_left, strafe_right, turn_left, turn_right, arc_left, arc_right
```

Each skill maps to an SE(2) velocity command. The low-level Go2 controller tracks that command in Isaac Lab, while the Mindcraft model learns to predict the next proprioceptive state, reward, done probability, value, and policy prior for each skill.
