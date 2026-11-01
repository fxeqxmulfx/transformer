"""First canonical validation crossing; final test never controls training.

Source: task154 CrossingTraining, extended to fresh matched task158 runs.
Unlike historical ResidualPairs.stable, this user-requested rule needs only
one validation observation at .99, with no training-accuracy requirement.
"""

import hashlib
import json
import math
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
    expected = declaration["cohort_sources"]
    amendment = HERE / "metadata_amendment.json"
    if amendment.exists():
        repair = json.loads(amendment.read_text())
        if repair["original_protocol_sha256"] != sha(HERE / "protocol.json"):
            raise ValueError("Metadata amendment belongs to another protocol")
        expected = repair["cohort_sources"]
    if declaration["engine_sources"] != code() or expected != sources():
        raise ValueError("Frozen task158 source changed; replay at its committed revision")
    return declaration


def same_data(actual, expected):
    """Require exact rows/counts/hashes; allow one ULP in a reduction diagnostic.

    P193 CPU and CUDA float64 means differ by one ULP despite identical rows.
    This diagnostic is never a label, training value or stopping measurement.
    The original protocol and the repair's exact source hashes are retained.
    """
    if actual.keys() != expected.keys():
        return False
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, dict):
            if not isinstance(reference, dict) or not same_data(value, reference):
                return False
        elif value != reference:
            if (key != "copy_last_accuracy" or not isinstance(value, float)
                    or not isinstance(reference, float) or not math.isfinite(value)
                    or not math.isfinite(reference)
                    or abs(value - reference) > max(math.ulp(value), math.ulp(reference))):
                return False
    return True


class CrossingTraining(ResidualTraining):
    def stopped(self):
        return bool(self.history) and self.history[-1]["validation"]["accuracy"] >= TARGET

    def finish(self):
        result = super().finish()
        if self.stopped():
            result["stop"]["reason"] = "validation99"
        self.run.finish(result)
        return result
