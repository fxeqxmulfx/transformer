"""Observe real archived moment states with one disposable CPU probe step.

From python/: uv run --locked python ../experiments/grokking_internals/compare_momentum.py.
Native CUDA training and every frozen earlier observer remain unchanged.
Only noninitial weights with actual optimizer and sampler states qualify.
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

from compare_objectives import CHOICES
from momentum_directions import measure

HERE = Path(__file__).resolve().parent


def identity():
    paths = [Path(__file__).resolve(), HERE / "momentum_directions.py", HERE / "compare_objectives.py",
             HERE / "probes/capture.py", HERE / "probes/gradients.py"]
    return {"lab": code(), "reader": {str(path.relative_to(HERE)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "torch": torch.__version__, "device": "cpu", "threads": 4,
            "lean_source_commit": "91bb895", "training_resumed": False}


def main():
    torch.set_num_threads(4)
    provenance = identity()
    path = HERE / "momentum_direction_results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["observer"] != provenance:
        raise RuntimeError("Reader changed: preserve existing results under their original version")
    result = {"observer": provenance, "selection": "same_pinned_weights; actual_noninitial_moment_and_sampler_states",
              "runs": [], "missing": [], "excluded": []}
    cached = {(run["study"], run["label"], record["step"]): record
              for run in previous.get("runs", []) for record in run["records"]}
    for study, label, steps in CHOICES:
        spec = dict(load(study).select([]))[label]
        root = study / "runs" / label
        manifest = json.loads((root / "experiment.json").read_text())
        trained = manifest["segments"][-1]["provenance"]["engine"]["code"]
        for name, digest in provenance["lab"].items():
            if name.startswith(("infrastructure/nn/", "infrastructure/optim/", "infrastructure/benchmarks/modular",
                                "infrastructure/benchmarks/samplers", "domain/training", "domain/optimizers",
                                "infrastructure/engine/eager", "infrastructure/engine/loop")) and trained.get(name) != digest:
                raise RuntimeError(f"Training model/task/update implementation differs: {name}")
        task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        run = {"study": study.name, "label": label, "corpus": task.summary(), "records": []}
        result["runs"].append(run)
        for step in steps:
            checkpoint = root / "probe_checkpoints" / f"step-{step:06d}.pt"
            if not checkpoint.exists():
                result["missing"].append({"study": study.name, "label": label, "step": step})
                continue
            data = checkpoint.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if step == 0:
                result["excluded"].append({"study": study.name, "label": label, "step": step,
                                           "reason": "seeded_initial_weights_have_no_archived_optimizer_or_sampler_state",
                                           "checkpoint_sha256": digest})
                continue
            record = cached.get((study.name, label, step))
            if record is not None and record["checkpoint_sha256"] != digest:
                raise RuntimeError(f"Previously observed checkpoint changed: {checkpoint}")
            if record is None:
                snapshot = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                if snapshot["step"] != step or snapshot.get("seeded_initialization"):
                    raise RuntimeError(f"Checkpoint origin/step mismatch: {checkpoint}")
                model.load_state_dict(snapshot["model"])
                started = time.monotonic()
                measurement = measure(model, task, spec, snapshot, spec.diagnostics.batch)
                record = {"step": step, "checkpoint_sha256": digest, "origin": "actual_training_checkpoint",
                          "measurement": measurement, "observation_seconds": time.monotonic() - started}
                population = measurement["populations"]["heldout_nonzero"]
                print(study.name, label, step, "accuracy", population["before"]["answer_accuracy"],
                      "heldout_answer_rate_derivative",
                      population["alignment"]["answer"]["total_algorithm"]["loss_rate_derivative"],
                      "finite_answer_CE_change", population["after"]["losses"]["answer"] - population["before"]["losses"]["answer"],
                      flush=True)
            run["records"].append(record)
            if identity() != provenance:
                raise RuntimeError("Observation source changed during measurement")
            replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
