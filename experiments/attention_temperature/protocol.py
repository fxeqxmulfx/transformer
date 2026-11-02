"""Freeze the isolated score-temperature comparison against the current-model pair."""

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
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def frozen():
    declaration = json.loads((HERE / "protocol.json").read_text())
    if declaration["engine_sources"] != code() or declaration["cohort_sources"] != sources():
        raise ValueError("Frozen control sources changed; replay its committed source")
    for name, digest in declaration["reference_files"].items():
        if sha(HERE.parent / name) != digest:
            raise ValueError("The declared reference comparison changed")
    return declaration
