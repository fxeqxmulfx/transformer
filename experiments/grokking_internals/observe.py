"""Measure every retained checkpoint on CPU; never resume or change training.

From python/: uv run --locked python ../experiments/grokking_internals/observe.py.
Records both studies, checkpoint/source hashes, initialized-model versus
training-checkpoint origin, and endpoint intervals. All model-building
source must match the training manifest. Observations resume only with
byte-identical probe code. Aggregate old curves cannot supply past weights.
"""

import hashlib
import io
import json
from pathlib import Path
import time

import torch

from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import replace

from probes.capture import collect
from probes.suite import Suite

HERE = Path(__file__).resolve().parent
STUDIES = (HERE, HERE.parent / "grokking_progress")


def identity():
    paths = [Path(__file__).resolve(), *sorted((HERE / "probes").glob("*.py"))]
    return {"lab": code(), "probes": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in paths}, "torch": torch.__version__, "observation_device": "cpu"}


class Observations:
    def __init__(self, study, label, experiment, provenance):
        self.study, self.label, self.experiment = study, label, experiment
        self.root = study / "runs" / label
        self.path = self.root / "internal_suite_v1.jsonl"
        self.meta = self.root / "internal_suite_v1_manifest.json"
        if self.meta.exists() and json.loads(self.meta.read_text())["observer"] != provenance:
            raise RuntimeError(f"Observer changed; retain old records under their original version: {self.path}")
        self.task = ModularTask(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        self.model = build_model(experiment.model, self.task.vocab, experiment.seeds.model)
        self.suite = Suite(experiment, self.task)
        manifest = json.loads((self.root / "experiment.json").read_text())
        trained = manifest["segments"][-1]["provenance"]["engine"]["code"]
        for name, digest in provenance["lab"].items():
            if name.startswith("infrastructure/nn/") and trained.get(name) != digest:
                raise RuntimeError(f"Training model implementation differs: {name}")
        if not self.meta.exists():
            replace(self.meta, json.dumps({"observer": provenance, "training_description": manifest["description"],
                "corpus": self.task.summary(), "choices": {"ridge": .001, "ablated_subspace_rank": 8,
                "gradient_batches": 8, "gradient_batch_size": 32}}, indent=2) + "\n")
        rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.last = rows[-1]["step"] if rows else -1
        checkpoint_folder = self.root / "probe_checkpoints"
        checkpoint_folder.mkdir(exist_ok=True)
        initial = checkpoint_folder / "step-000000.pt"
        if not initial.exists():
            torch.save({"step": 0, "model": self.model.state_dict(), "seeded_initialization": True}, initial)
        if rows:
            previous = torch.load(checkpoint_folder / f"step-{self.last:06d}.pt", map_location="cpu", weights_only=True)
            self.model.load_state_dict(previous["model"])
            logits, features = collect(self.model, self.suite.orbits.inputs, experiment.diagnostics.batch)
            self.suite.previous = (self.last, logits)

    def next(self):
        snapshots = sorted((self.root / "probe_checkpoints").glob("step-*.pt"))
        return next((path for path in snapshots if int(path.stem.split("-")[1]) > self.last), None)

    def measure(self, checkpoint):
        data = checkpoint.read_bytes()
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model"])
        result = self.suite.observe(self.model, saved["step"])
        result.update(checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                      checkpoint_origin="seeded_initial_model" if saved.get("seeded_initialization") else "atomic_training_checkpoint",
                      observation_device="cpu", training_device=self.experiment.execution.device)
        with self.path.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        self.last = saved["step"]
        print(self.study.name, self.label, self.last, "accuracy", result["output"]["raw_heldout_accuracy"],
              "probe", result["features"]["blocks.1"]["linear_probe"]["heldout_accuracy"],
              "seconds", round(result["observation_seconds"], 2), flush=True)


def main():
    torch.set_num_threads(4)
    provenance = identity()
    active = {}
    while True:
        declared = {(study, label): experiment
                    for study in STUDIES for label, experiment in load(study).select([])}
        changed = False
        for (study, label), experiment in declared.items():
            if not (study / "runs" / label / "experiment.json").exists():
                continue
            key = (study, label)
            if key not in active:
                active[key] = Observations(study, label, experiment, provenance)
            observer = active[key]
            checkpoint = observer.next()
            if checkpoint is not None:
                if identity() != provenance:
                    raise RuntimeError("Observer source changed during observation")
                observer.measure(checkpoint)
                changed = True
        if len(active) == len(declared) and all(observer.last == observer.experiment.budget.updates
                                               for observer in active.values()):
            return
        if not changed:
            time.sleep(5)


if __name__ == "__main__":
    main()
