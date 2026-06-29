#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard replay JSONL files into one replay buffer.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], type=Path)
    parser.add_argument("--source-glob", action="append", default=[])
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_existing_hashes(args.output)
    offsets: dict[Path, int] = {}
    while True:
        sources = _sources(args.source, args.source_glob, args.output)
        appended = 0
        with args.output.open("a", encoding="utf-8") as out:
            for source in sources:
                new_offset, records = _read_new_records(source, offsets.get(source, 0))
                offsets[source] = new_offset
                for record in records:
                    line = json.dumps(record, sort_keys=True)
                    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    out.write(line + "\n")
                    appended += 1
        if appended:
            print(f"merged {appended} transitions from {len(sources)} source(s) into {args.output}", flush=True)
        if not args.follow:
            break
        time.sleep(max(0.1, args.poll_interval))


def _sources(explicit: list[Path], patterns: list[str], output: Path) -> list[Path]:
    paths = set(explicit)
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    return sorted(path for path in paths if path.exists() and path.resolve() != output.resolve())


def _load_existing_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        canonical = json.dumps(record, sort_keys=True)
        seen.add(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    return seen


def _read_new_records(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return offset, records
    if path.stat().st_size < offset:
        offset = 0
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                offset = f.tell()
                break
            stripped = line.strip()
            if not stripped:
                offset = f.tell()
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                offset = line_start
                break
            offset = f.tell()
    return offset, records


if __name__ == "__main__":
    main()
