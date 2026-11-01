"""Task154: pin both parent states, continue unchanged, stop at 99% validation.

Uses task145's ResidualTraining/ResidualCompiledStepper. The only stopping
deviation is the user's requested first canonical validation crossing of .99.
The parent cohort, its checkpoints and its earlier final-test results stay intact.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from lab.domain.experiment import require_continuation
from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.residual import ResidualCompiledStepper, ResidualTraining
from lab.infrastructure.loader import load
from lab.infrastructure.provenance import code, provenance
from lab.infrastructure.store import RunDirectory, replace

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "residual_generalization"
PROTOCOL = HERE / "protocol.json"
PHASE = HERE / "runs/phase.json"
TARGET = .99


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    replace(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources():
    return {path.name: sha(path) for path in sorted(HERE.glob("*.py"))}


def same_state(left, right):
    """Exact recursive comparison, including optimizer clocks and sampler state."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("The continuation protocol is already frozen")
    old = json.loads((PARENT / "protocol.json").read_text())
    old_results = json.loads((PARENT / "results.json").read_text())
    current = code()
    changed = {name for name, digest in old["engine_sources"].items() if current.get(name) != digest}
    # These reviewed changes only register the separate task151 witness models.
    reviewed = {"domain/experiment.py", "dsl.py", "infrastructure/benchmarks/__init__.py",
                "infrastructure/engine/__init__.py", "infrastructure/nn/__init__.py"}
    if changed != reviewed:
        raise ValueError(f"Unexpected parent engine changes: {changed}")
    cases = []
    for label, spec in load(HERE).experiments.items():
        parent_label = f"mechanism-s{spec.seeds.model}-scaled"
        declaration = next(case for case in old["cases"] if case["label"] == parent_label)
        result = next(case for case in old_results["cases"] if case["label"] == parent_label)
        parent = RunDirectory(PARENT / "runs" / parent_label)
        require_continuation(parent.description(), spec)
        if result["step"] != 23000 or sha(parent.path / "checkpoint.pt") != result["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint is not the declared 23k endpoint")
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        if task.metadata != declaration["task"]:
            raise ValueError("Parent split fingerprints changed")
        cases.append({"label": label, "parent_label": parent_label, "from_step": 23000,
            "experiment": describe(spec), "task": task.metadata,
            "parent_files": {p.name: sha(p) for p in sorted(parent.path.iterdir()) if p.is_file()}})
    write(PROTOCOL, {"frozen_utc": datetime.now(timezone.utc).isoformat(),
        "requested_stop": "first canonical validation accuracy >=0.99, every1000 updates; current endpoint",
        "target_accuracy": TARGET, "cases": cases, "engine_sources": current,
        "cohort_sources": sources(), "parent_protocol_sha256": sha(PARENT / "protocol.json"),
        "parent_results_sha256": sha(PARENT / "results.json"),
        "parent_scoring_sha256": sha(PARENT / "analyze.py"),
        "reviewed_parent_dispatch_changes": sorted(changed),
        "runtime": "same Python3.15/Torch2.15, CUDA fullgraph max-autotune with math SDPA",
        "selection": "no learning-rate, architecture, optimizer, split, batch-order or seed changes",
        "scope": "two continuations, not independent new seeds; final test reused from the previously reported cohort; not used for stopping",
        "comparison": "paired ordinary controls' first validation>=99% crossing, not their later99.9% stable stop"})
    print("Frozen both complete 23k parent states and unchanged arithmetic sources", flush=True)


class CrossingTraining(ResidualTraining):
    def stopped(self):
        return bool(self.history) and self.history[-1]["validation"]["accuracy"] >= TARGET

    def measure(self):
        return {**super().measure(), "depth_queries": []}

    def finish(self):
        result = super().finish()
        if self.stopped():
            result["stop"]["reason"] = "validation99"
        self.run.finish(result)
        return result


def run():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["engine_sources"] != code() or protocol["cohort_sources"] != sources():
        raise ValueError("Frozen continuation sources changed")
    if sha(PARENT / "protocol.json") != protocol["parent_protocol_sha256"] or sha(PARENT / "results.json") != protocol["parent_results_sha256"]:
        raise ValueError("Frozen parent reports changed")
    if PHASE.exists():
        raise FileExistsError("This continuation phase already started; inspect its retained states")
    phase = {"status": "learning", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    write(PHASE, phase)
    study = load(HERE)
    for case in protocol["cases"]:
        label = case["label"]
        record = {"label": label, "status": "preparing"}
        phase["cases"].append(record)
        write(PHASE, phase)
        start = time.monotonic()
        parent = PARENT / "runs" / case["parent_label"]
        for name, digest in case["parent_files"].items():
            if sha(parent / name) != digest:
                raise ValueError(f"Parent state changed: {name}")
        run = RunDirectory(HERE / "runs" / label)
        shutil.copytree(parent, run.path)
        spec = study.experiments[label]
        require_continuation(run.description(), spec)
        source = run.checkpoint("cpu")
        run.begin(describe(spec), provenance(spec.execution.device), study.source)
        trainer = CrossingTraining(spec, run,
            lambda row: print(json.dumps({"label": label, **row}, allow_nan=False), flush=True),
            ResidualCompiledStepper)
        checks = {"model": same_state(source["model"], trainer.model.state_dict()),
                  "optimizer": same_state(source["optimizer"], trainer.stepper.state_dict()),
                  "sampler": all(same_state(source[key], value) for key, value in trainer.sampler.state().items()),
                  "step": trainer.completed == source["step"], "examples": trainer.seen == source["examples_seen"]}
        record["restoration_after_compile"] = checks
        if not all(checks.values()):
            raise AssertionError(f"Compilation or restoration changed the parent state: {checks}")
        record["preparation_seconds"] = time.monotonic() - start
        record["status"] = "learning"
        write(PHASE, phase)
        result = trainer()
        record.update(status=result["stop"]["reason"], stop=result["stop"],
                      elapsed_seconds=time.monotonic() - start,
                      checkpoint_sha256=sha(run.path / "checkpoint.pt"))
        write(PHASE, phase)
        print(json.dumps(record), flush=True)
        del trainer
        torch.cuda.empty_cache()
    phase["status"] = "learning_closed"
    phase["closed_utc"] = datetime.now(timezone.utc).isoformat()
    write(PHASE, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    options = parser.parse_args()
    (freeze if options.action == "freeze" else run)()
