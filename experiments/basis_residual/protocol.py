"""Frozen task160 sources, first99% sequence endpoints and original sample overlap."""

import hashlib
import json
from pathlib import Path

from lab.infrastructure.provenance import code
from lab.infrastructure.store import replace

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    replace(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources():
    return {path.name: sha(path) for path in sorted(HERE.glob("*.py"))}


def frozen():
    declaration = json.loads((HERE / "protocol.json").read_text())
    if declaration["engine_sources"] != code() or declaration["cohort_sources"] != sources():
        raise ValueError("Frozen Basis source changed; replay at its committed revision")
    return declaration


def novel_examples(task, rows):
    """Original generator identity; parity uses prompts, never teacher-forced answers."""
    training = {task.generator.key(example) for example in task.splits["train"].examples}
    return tuple(example for example in rows.examples if task.generator.key(example) not in training)


def overlap(task):
    return {name: {"examples": len(rows), "unique_inputs": len({task.generator.key(e) for e in rows.examples}),
                   "novel_against_training": len(novel_examples(task, rows))}
            for name, rows in task.splits.items() if name != "train"}
