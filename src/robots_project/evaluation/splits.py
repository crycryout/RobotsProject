"""Check disjoint scene/seed identities before evaluation or checkpoint selection."""

import json
from pathlib import Path


def validate_splits(directory: Path):
    seen_ids, seen_seeds = set(), set()
    result = {}
    for path in sorted(directory.glob("*.jsonl")):
        split = path.stem
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty split: {split}")
        for row in rows:
            if row["split"] != split or row["episode_id"] in seen_ids or row["seed"] in seen_seeds:
                raise ValueError("Train/validation/test episode IDs and seeds must be disjoint")
            seen_ids.add(row["episode_id"])
            seen_seeds.add(row["seed"])
        result[split] = len(rows)
    if not {"train", "validation", "test"}.issubset(result):
        raise ValueError("Required train, validation, and test manifests missing")
    return result
