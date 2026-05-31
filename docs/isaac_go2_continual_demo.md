# Isaac Lab Unitree Go2 Continual-Learning Demo

This demo reuses the continual-learning idea from Mindcraft in a robotics setting:

- Isaac Lab supplies the default Unitree Go2 velocity-tracking scene.
- A low-level RSL-RL policy turns velocity commands into joint actions.
- The Mindcraft world model learns action-conditioned robot dynamics online from replay.
- A persistent skill library and model-based selector choose high-level locomotion skills.

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

To record a short headless mp4 aimed at the first Go2:

```sh
cd /home/trantor/Research/mindcraft-multi-wm
PYTHONPATH=$PWD /home/trantor/Research/computational_robotics/IsaacLab/isaaclab.sh \
  -p scripts/isaac_go2_continual_demo.py \
  --headless \
  --video \
  --video-length 900 \
  --video-dir runs/isaac_go2/videos \
  --num_envs 1 \
  --episodes 1 \
  --skills-per-episode 32
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
- `videos/*.mp4`: optional Isaac Lab renderings when `--video` is set.

## What Is Being Learned

The high-level action space is a discrete robotics skill vocabulary:

```text
stand, walk_forward_slow, walk_forward, walk_backward,
strafe_left, strafe_right, turn_left, turn_right, arc_left, arc_right
```

Each skill maps to an SE(2) velocity command. The low-level Go2 controller tracks that command in Isaac Lab, while the Mindcraft model learns to predict the next proprioceptive state, reward, done probability, value, and policy prior for each skill.
