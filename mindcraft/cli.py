from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from mindcraft.replay import ReplayBuffer
from mindcraft.training_logs import TensorboardLogger, append_training_metrics
from mindcraft.world_model import WorldModelTrainer, sample_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mindcraft world-model training tools")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-replay", help="Train the action-conditioned world model from JSONL replay")
    train.add_argument("--storage-dir", default="runs/default")
    train.add_argument("--replay-file", default=None, help="Defaults to <storage-dir>/experience.jsonl")
    train.add_argument("--batches", type=int, default=500, help="Use <=0 to train forever with --follow")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--sequence-length", type=int, default=8)
    train.add_argument("--replay-capacity", type=int, default=50_000)
    train.add_argument("--frontier-sampling", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--hindsight-relabeling", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    train.add_argument("--model-dim", type=int, default=128)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--latent-dim", type=int, default=16)
    train.add_argument("--ssm-state-dim", type=int, default=16)
    train.add_argument("--ensemble-size", type=int, default=3)
    train.add_argument("--lora-rank", type=int, default=4)
    train.add_argument("--freeze-base-for-lora", action="store_true")
    train.add_argument("--lr", type=float, default=3.0e-4)
    train.add_argument("--gamma", type=float, default=0.97)
    train.add_argument("--checkpoint-name", default="world_model.pt")
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument("--follow", action="store_true", help="Refresh replay and keep training as more data arrives")
    train.add_argument("--poll-interval", type=float, default=2.0)
    train.add_argument("--print-every", type=int, default=25)
    train.add_argument("--phase", default="replay")
    train.add_argument("--tensorboard", action="store_true", help="Write TensorBoard events under storage-dir")
    train.add_argument("--torch-threads", type=int, default=None)

    info = sub.add_parser("device-info", help="Print the torch runtime selected by the world model")
    info.add_argument("--torch-threads", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "train-replay":
        _set_torch_threads(args.torch_threads)
        _train_replay(args)
    elif args.command == "device-info":
        _set_torch_threads(args.torch_threads)
        print(sample_device())


def _train_replay(args: argparse.Namespace) -> None:
    storage_dir = Path(args.storage_dir)
    replay_path = Path(args.replay_file) if args.replay_file else storage_dir / "experience.jsonl"
    replay = ReplayBuffer(
        replay_path,
        capacity=args.replay_capacity,
        hindsight_relabeling=args.hindsight_relabeling,
        frontier_sampling=args.frontier_sampling,
    )
    trainer = WorldModelTrainer(
        storage_dir,
        d_model=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        latent_dim=args.latent_dim,
        ssm_state_dim=args.ssm_state_dim,
        ensemble_size=args.ensemble_size,
        lora_rank=args.lora_rank,
        freeze_base_for_lora=args.freeze_base_for_lora,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
    )
    rng = random.Random(args.seed)
    tensorboard = TensorboardLogger(storage_dir, enabled=args.tensorboard)
    latest = None
    idx = 0
    try:
        while args.batches <= 0 or idx < args.batches:
            if args.follow:
                replay.refresh()
            if not replay.can_sample_sequence(args.sequence_length):
                if not args.follow:
                    raise SystemExit(
                        f"not enough per-agent replay for sequence length {args.sequence_length}; "
                        f"loaded {len(replay)} transitions from {replay_path}"
                    )
                print(
                    f"waiting for per-agent replay sequence length {args.sequence_length}; "
                    f"loaded {len(replay)} transitions from {replay_path}"
                )
                time.sleep(max(0.1, args.poll_interval))
                continue

            seqs = replay.sample_sequences(args.batch_size, args.sequence_length, rng)
            val_seqs = replay.sample_validation_sequences(max(1, args.batch_size // 4), args.sequence_length, rng)
            latest = trainer.train_batches(seqs, validation_sequences=val_seqs)
            if latest is None:
                time.sleep(max(0.1, args.poll_interval))
                continue

            idx += 1
            append_training_metrics(storage_dir, metrics=latest, replay_size=len(replay), phase=args.phase)
            tensorboard.log_world_model(metrics=latest, replay_size=len(replay), phase=args.phase)

            if args.print_every > 0 and (idx == 1 or idx % args.print_every == 0):
                print(
                    f"batch {idx}/{args.batches if args.batches > 0 else 'inf'} "
                    f"train_step={latest.train_step} replay={len(replay)} "
                    f"loss={latest.loss:.4f} obs={latest.obs_loss:.4f} "
                    f"jepa={latest.jepa_loss:.4f} reward={latest.reward_loss:.4f} "
                    f"value={latest.value_loss:.4f} policy={latest.policy_loss:.4f} "
                    f"code_usage={latest.code_usage:.4f}"
                )

            if args.checkpoint_every > 0 and latest.train_step % args.checkpoint_every == 0:
                trainer.save()
                print(f"checkpointed world model at train_step={latest.train_step} to {trainer.checkpoint_path}")
    finally:
        tensorboard.close()

    if latest is None:
        raise SystemExit("no trainable replay batches were sampled")
    trainer.save()
    print(f"saved world model to {trainer.checkpoint_path}; train_step={trainer.train_step} loss={trainer.last_loss:.4f}")


def _set_torch_threads(count: int | None) -> None:
    if count is None:
        return
    import torch

    torch.set_num_threads(max(1, count))


if __name__ == "__main__":
    main()
