"""Evaluate every preserved checkpoint of all completed grokking runs.

From python/: uv run --locked python ../experiments/grokking_internals/compare_geometry_all.py.
Distinct protocol from the pinned objective comparison: preserve its reader
and results, inspect all currently retained weights, require the full-budget
endpoint, and record archive coverage without reconstructing absent weights.
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

from geometry_certificates import measure
from probes.capture import evaluating, forward

HERE = Path(__file__).resolve().parent
STUDIES = (HERE, HERE.parent / "grokking_progress")


def identity():
    paths = [Path(__file__).resolve(), HERE / "geometry_certificates.py", HERE / "probes" / "capture.py"]
    return {"lab": code(), "reader": {str(path.relative_to(HERE)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "torch": torch.__version__, "device": "cpu", "threads": 4,
            "lean_source_commit": "3661a44", "arithmetic": "exact_binary_float_ratios_for_certificate_counts"}


def main():
    torch.set_num_threads(4)
    provenance = identity()
    path = HERE / "geometry_all_results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["observer"] != provenance:
        raise RuntimeError("Reader changed: preserve existing results under their original version")
    result = {"observer": provenance, "selection": "all_preserved_weights_of_completed_full_budget_runs",
              "runs": []}
    cached = {(run["study"], run["label"], record["step"]): record
              for run in previous.get("runs", []) for record in run["records"]}
    for study in STUDIES:
        for label, spec in load(study).select([]):
            root = study / "runs" / label
            manifest = json.loads((root / "experiment.json").read_text())
            trained = manifest["segments"][-1]["provenance"]["engine"]["code"]
            for name, digest in provenance["lab"].items():
                if name.startswith(("infrastructure/nn/", "infrastructure/benchmarks/modular")) and trained.get(name) != digest:
                    raise RuntimeError(f"Training model/task implementation differs: {name}")
            checkpoints = sorted((root / "probe_checkpoints").glob("step-*.pt"))
            steps = [int(checkpoint.stem.split("-")[1]) for checkpoint in checkpoints]
            if not steps or steps[0] != 0 or steps[-1] != spec.budget.updates:
                raise RuntimeError(f"Completed full-budget archive required: {root}")
            task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
            model = build_model(spec.model, task.vocab, spec.seeds.model)
            orbit = GrokkingObserver(spec.diagnostics, task)
            run = {"study": study.name, "label": label, "corpus": task.summary(),
                   "budget_updates": spec.budget.updates, "available_steps": steps,
                   "selection_scope": "available_weights_only; gaps_remain_unobserved", "records": []}
            result["runs"].append(run)
            for checkpoint, step in zip(checkpoints, steps):
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
