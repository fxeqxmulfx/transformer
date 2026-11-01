"""First canonical validation crossing; final test never controls training.

Source: task154 CrossingTraining, extended to fresh matched task158 runs.
Unlike historical ResidualPairs.stable, this user-requested rule needs only
one validation observation at .99, with no training-accuracy requirement.
"""

import hashlib
import json
from pathlib import Path

from lab.infrastructure.engine.residual import ResidualTraining
from lab.infrastructure.provenance import code
from lab.infrastructure.store import replace

HERE = Path(__file__).resolve().parent
TARGET = .99


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
        raise ValueError("Frozen task158 source changed; replay at its committed revision")
    return declaration


class CrossingTraining(ResidualTraining):
    def stopped(self):
        return bool(self.history) and self.history[-1]["validation"]["accuracy"] >= TARGET

    def finish(self):
        result = super().finish()
        if self.stopped():
            result["stop"]["reason"] = "validation99"
        self.run.finish(result)
        return result
