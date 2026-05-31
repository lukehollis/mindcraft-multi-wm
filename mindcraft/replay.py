from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable

from mindcraft.schemas import Transition
from mindcraft.progression import replay_priority, transition_event_bucket


class ReplayBuffer:
    def __init__(self, path: Path, capacity: int = 50_000):
        self.path = path
        self.capacity = capacity
        self.items: deque[Transition] = deque(maxlen=capacity)
        self._offset = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.refresh()

    def __len__(self) -> int:
        return len(self.items)

    def append(self, transition: Transition) -> None:
        self.items.append(transition)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(transition.to_jsonable(), sort_keys=True) + "\n")
            self._offset = f.tell()

    def refresh(self) -> int:
        """Load transitions appended by another process since the last read."""
        if not self.path.exists():
            return 0
        if self.path.stat().st_size < self._offset:
            self.items.clear()
            self._offset = 0
        return self._load_new()

    def sample_sequences(self, batch_size: int, sequence_length: int, rng: random.Random) -> list[list[Transition]]:
        if batch_size <= 0 or sequence_length <= 0:
            return []
        windows = self._sequence_windows(sequence_length, holdout=False)
        if not windows:
            windows = self._sequence_windows(sequence_length)
        if not windows:
            return []
        return _sample_stratified(windows, batch_size, rng)

    def sample_validation_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        rng: random.Random,
    ) -> list[list[Transition]]:
        if batch_size <= 0 or sequence_length <= 0:
            return []
        windows = self._sequence_windows(sequence_length, holdout=True)
        if not windows:
            return []
        return _sample_stratified(windows, batch_size, rng)

    def can_sample_sequence(self, sequence_length: int) -> bool:
        return bool(self._sequence_windows(sequence_length))

    def tail(self, count: int) -> Iterable[Transition]:
        return list(self.items)[-count:]

    def _load_new(self) -> int:
        loaded = 0
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._offset)
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(Transition.from_jsonable(json.loads(line)))
                except JSONDecodeError:
                    self._offset = line_start
                    break
                loaded += 1
                self._offset = f.tell()
        return loaded

    def _sequence_windows(self, sequence_length: int, holdout: bool | None = None) -> list[list[Transition]]:
        if len(self.items) < sequence_length:
            return []
        trajectories: dict[tuple[int, str], list[Transition]] = defaultdict(list)
        for transition in self.items:
            trajectories[(transition.episode, transition.agent)].append(transition)

        windows: list[list[Transition]] = []
        window_index = 0
        for trajectory in trajectories.values():
            if len(trajectory) < sequence_length:
                continue
            max_start = len(trajectory) - sequence_length
            for start in range(max_start + 1):
                is_holdout = window_index % 5 == 4
                window_index += 1
                if holdout is None or holdout == is_holdout:
                    windows.append(trajectory[start : start + sequence_length])
        return windows


def _sample_stratified(
    windows: list[list[Transition]],
    batch_size: int,
    rng: random.Random,
) -> list[list[Transition]]:
    buckets: dict[tuple[str, str, str, bool], list[list[Transition]]] = defaultdict(list)
    bucket_weights: dict[tuple[str, str, str, bool], float] = {}
    for window in windows:
        last = window[-1]
        key = (transition_event_bucket(last), last.agent, last.skill, last.result.success)
        buckets[key].append(window)
        bucket_weights[key] = max(bucket_weights.get(key, 0.0), replay_priority(last))
    keys = list(buckets)
    rng.shuffle(keys)
    sequences: list[list[Transition]] = []
    for key in sorted(keys, key=lambda current: (-bucket_weights[current], current)):
        sequences.append(_weighted_window_choice(buckets[key], rng))
        if len(sequences) >= batch_size:
            rng.shuffle(sequences)
            return sequences
    while len(sequences) < batch_size:
        key = _weighted_key_choice(keys, bucket_weights, rng)
        sequences.append(_weighted_window_choice(buckets[key], rng))
    rng.shuffle(sequences)
    return sequences


def _weighted_key_choice(
    keys: list[tuple[str, str, str, bool]],
    weights: dict[tuple[str, str, str, bool], float],
    rng: random.Random,
) -> tuple[str, str, str, bool]:
    total = sum(max(0.05, weights[key]) for key in keys)
    target = rng.random() * total
    running = 0.0
    for key in keys:
        running += max(0.05, weights[key])
        if running >= target:
            return key
    return keys[-1]


def _weighted_window_choice(windows: list[list[Transition]], rng: random.Random) -> list[Transition]:
    total = sum(replay_priority(window[-1]) for window in windows)
    target = rng.random() * total
    running = 0.0
    for window in windows:
        running += replay_priority(window[-1])
        if running >= target:
            return list(window)
    return list(windows[-1])
