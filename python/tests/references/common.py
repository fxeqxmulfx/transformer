"""Measurement helpers; these contain no model, training or optimizer implementation."""

import hashlib
import json


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def hashes(state):
    return {name: sha(value) for name, value in state.items()}


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def untimed(rows):
    return [{key: value for key, value in row.items() if key not in ("training_seconds", "wall_seconds")}
            for row in rows]
