"""Read selected immutable checkpoints and test the shared-EOS hypothesis.

From python/: uv run --locked python ../experiments/grokking_internals/compare_objectives.py.
Pinned checkpoint choices are diagnostic observations, not training variants.
Missing weights are recorded and can be filled by rerunning this reader.
The six-measurement worker has its own unchanged source identity.
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

from objective_components import measure

HERE = Path(__file__).resolve().parent
CHOICES = ((HERE, "gptmini-seed1", (0, 1000, 30000, 33000, 34000, 35000, 36000, 40000, 150000)),
           (HERE.parent / "grokking_progress", "gptmini-seed2", (0, 5000, 30000, 35000, 40000, 150000)),
           (HERE.parent / "grokking_progress", "gptmini-seed3", (0, 5000, 30000, 35000, 40000, 150000)),
           (HERE.parent / "grokking_progress", "reference-fraction20", (0, 5000, 30000, 35000, 40000, 150000)))


def identity():
    paths = [Path(__file__).resolve(), HERE / "objective_components.py",
             HERE / "probes" / "capture.py", HERE / "probes" / "gradients.py"]
    return {"lab": code(), "reader": {str(path.relative_to(HERE)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "torch": torch.__version__, "device": "cpu", "batches": 8, "batch_size": 32}


def main():
    torch.set_num_threads(4)
    provenance = identity()
    path = HERE / "objective_component_results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["observer"] != provenance:
        raise RuntimeError("Reader changed: preserve the existing result under its original version")
    result = {"observer": provenance, "selection": "pinned_checkpoint_steps; available_weights_only",
              "runs": [], "missing": []}
    cached = {(run["study"], run["label"], record["step"]): record
              for run in previous.get("runs", []) for record in run["records"]}
    for study, label, steps in CHOICES:
        spec = dict(load(study).select([]))[label]
        root = study / "runs" / label
        manifest = json.loads((root / "experiment.json").read_text())
        trained = manifest["segments"][-1]["provenance"]["engine"]["code"]
        for name, digest in provenance["lab"].items():
            if name.startswith(("infrastructure/nn/", "infrastructure/benchmarks/modular")) and trained.get(name) != digest:
                raise RuntimeError(f"Training forward or objective implementation differs: {name}")
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
            record = cached.get((study.name, label, step))
            if record is not None and record["checkpoint_sha256"] != digest:
                raise RuntimeError(f"Previously measured checkpoint changed: {checkpoint}")
            if record is None:
                snapshot = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                if snapshot["step"] != step or bool(snapshot.get("seeded_initialization")) != (step == 0):
                    raise RuntimeError(f"Checkpoint origin or step mismatch: {checkpoint}")
                model.load_state_dict(snapshot["model"])
                started = time.monotonic()
                measurement = measure(model, task)
                record = {"step": step, "checkpoint_sha256": digest,
                          "origin": "seeded_initialization" if step == 0 else "actual_training_checkpoint",
                          "measurement": measurement, "observation_seconds": time.monotonic() - started}
                print(study.name, label, step,
                      {key: value["train_heldout_mean_gradient_cosine"]
                       for key, value in measurement["groups"]["all"]["components"].items()}, flush=True)
            run["records"].append(record)
            if identity() != provenance:
                raise RuntimeError("Observation source changed during measurement")
            replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
