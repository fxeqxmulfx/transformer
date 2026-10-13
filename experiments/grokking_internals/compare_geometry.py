"""Read pinned immutable weights and evaluate the Lean cleanup criterion.

From python/: uv run --locked python ../experiments/grokking_internals/compare_geometry.py.
The existing six-probe and objective readers/results retain their original
identities. This reader never resumes training or reconstructs missing weights.
"""

import hashlib
import io
import json
from pathlib import Path
import time

import torch

from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.engine.grokking import GrokkingObserver
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import replace

from compare_objectives import CHOICES
from geometry_certificates import measure
from probes.capture import evaluating, forward

HERE = Path(__file__).resolve().parent


def identity():
    paths = [Path(__file__).resolve(), HERE / "geometry_certificates.py",
             HERE / "compare_objectives.py", HERE / "probes" / "capture.py"]
    return {"lab": code(), "reader": {str(path.relative_to(HERE)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "torch": torch.__version__, "device": "cpu", "threads": 4,
            "lean_source_commit": "3661a44", "arithmetic": "exact_binary_float_ratios_for_certificate_counts"}


def main():
    torch.set_num_threads(4)
    provenance = identity()
    path = HERE / "geometry_certificate_results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["observer"] != provenance:
        raise RuntimeError("Reader changed: preserve existing results under their original version")
    result = {"observer": provenance, "selection": "same_pinned_steps_as_objective_components; available_weights_only",
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
                raise RuntimeError(f"Training model/task implementation differs: {name}")
        task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        orbit = GrokkingObserver(spec.diagnostics, task)
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
                raise RuntimeError(f"Previously observed checkpoint changed: {checkpoint}")
            if record is None:
                saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                if saved["step"] != step or bool(saved.get("seeded_initialization")) != (step == 0):
                    raise RuntimeError(f"Checkpoint origin/step mismatch: {checkpoint}")
                model.load_state_dict(saved["model"])
                started = time.monotonic()
                with evaluating(model):
                    logits = forward(model, orbit.inputs, spec.diagnostics.batch)
                measurement = measure(logits.reshape(orbit.prime, orbit.prime - 1, orbit.vocab),
                                      orbit.heldout, orbit.targets, spec.diagnostics.energy_floor)
                record = {"step": step, "checkpoint_sha256": digest,
                          "origin": "seeded_initialization" if step == 0 else "actual_training_checkpoint",
                          "rows_sha256": orbit.fingerprint, "measurement": measurement,
                          "observation_seconds": time.monotonic() - started}
                print(study.name, label, step, "accuracy", measurement["raw_nonzero_accuracy"],
                      "certified", measurement["certified_fraction"],
                      "safety", measurement["mean_signed_safety"], flush=True)
            run["records"].append(record)
            if identity() != provenance:
                raise RuntimeError("Observation source changed during measurement")
            replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
