"""One frozen case, ordinary lab state and a wall-time-censored current endpoint."""

import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

import torch

from lab.domain.experiment import require_continuation
from lab.domain.spec import describe
from lab.infrastructure.engine.residual import ResidualCompiledStepper, ResidualTraining
from lab.infrastructure.loader import load
from lab.infrastructure.provenance import provenance
from lab.infrastructure.store import RunDirectory
from phase import require_budget

HERE = Path(__file__).resolve().parent


class BoundedTraining(ResidualTraining):
    """Time is checked at canonical observations; interrupted state stays resumable."""
    def __init__(self, *args, allowance, **kwargs):
        self.runtime_deadline = float("inf")
        self.runtime_expired = False
        start = time.monotonic()
        super().__init__(*args, **kwargs)
        self.preparation_seconds = time.monotonic() - start
        self.runtime_deadline = time.monotonic() + min(allowance, max(0, require_budget(180)))

    def stopped(self):
        return super().stopped() or self.runtime_expired

    def observe(self, step, probe):
        super().observe(step, probe)
        if not probe and time.monotonic() >= self.runtime_deadline:
            self.runtime_expired = True

    def measure(self):
        result = super().measure()
        queries = []
        group = self.stepper.optimizer.param_groups[0]
        beta = group["betas"][0]
        for mixer in self.model.residual_mixers:
            query = mixer.query.detach()
            base = self.stepper.optimizer.state[mixer.query]["base"]
            average = torch.lerp(query, base, 1 - 1 / beta)
            queries.append({"query_rms": float(query.square().mean().sqrt()),
                "base_rms": float(base.square().mean().sqrt()),
                "average_rms": float(average.square().mean().sqrt()),
                "coordinate_scale": float(getattr(mixer, "query_scale", 1.))})
        result["depth_queries"] = queries
        return result

    def finish(self):
        result = super().finish()
        if self.runtime_expired and not self.experiment.benchmark.stable(self.history):
            result["stop"]["reason"] = "wall_budget_censored"
            result["requested_updates"] = result["updates"]
            result["updates"] = result["stop"]["step"]
        result["preparation_seconds"] = self.preparation_seconds
        self.run.finish(result)
        return result


def main():
    label = sys.argv[1]
    protocol = json.loads((HERE / "protocol.json").read_text())
    declaration = next(case for case in protocol["cases"] if case["label"] == label)
    continuation = len(sys.argv) == 3
    allowance = float(sys.argv[2]) if continuation else declaration["learning_wall_seconds"]
    require_budget(180)
    study = load(HERE)
    spec = study.experiments[label]
    run = RunDirectory(HERE / "runs" / label)
    previous = run.result()
    if run.description() is not None:
        require_continuation(run.description(), spec)
        if not continuation or previous is None or previous["stop"]["reason"] != "wall_budget_censored":
            raise FileExistsError("Only the declared continuation round can extend a time-censored case")
        source = run.path / "checkpoint.pt"
        destination = run.path / "first-phase.pt"
        if destination.exists():
            raise FileExistsError("The declared continuation already began")
        shutil.copyfile(source, destination)
    elif continuation:
        raise FileNotFoundError("Continuation requires the complete first-phase state")
    run.begin(describe(spec), provenance(spec.execution.device), study.source)
    trainer = BoundedTraining(spec, run,
        lambda row: print(json.dumps({"label": label, **row}, allow_nan=False), flush=True),
        ResidualCompiledStepper, allowance=allowance)
    if not continuation:
        trainer.save(0)
    initial = run.path / "checkpoint.pt"
    initial_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
    result = trainer()
    result["initial_checkpoint_sha256"] = initial_hash if previous is None else previous["initial_checkpoint_sha256"]
    if continuation:
        result["continuation"] = {"source_checkpoint_sha256": initial_hash,
                                  "from_step": previous["stop"]["step"], "allowance_seconds": allowance}
    result["final_checkpoint_sha256"] = hashlib.sha256(initial.read_bytes()).hexdigest()
    run.finish(result)
    print(json.dumps({"label": label, "stop": result["stop"]}), flush=True)


if __name__ == "__main__":
    main()
