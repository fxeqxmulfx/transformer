"""Read fixed-displacement CE/HVP curves at the archived native states.

From python/: uv run --locked python ../experiments/grokking_internals/compare_curvature.py.
Source: the CPU protocol and completed observations at 444b4ad; the interval
descent laws and initial-curvature counterexamples at cb38c97. The original
optimizer, sampler, losses and model are preserved. Interpolations and one
disposable CPU step are diagnostics, not continuations of CUDA training.
"""

import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch

from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import replace

from compare_momentum import identity as momentum_identity
from compare_objectives import CHOICES
from curvature_profiles import measure

HERE = Path(__file__).resolve().parent
MOMENTUM_SHA256 = "cace3f218081776d49ce5184154e3a3b0ee27bf1473cd45037fb17f6db1d3c5c"


def identity():
    paths = [Path(__file__).resolve(), HERE / "curvature_profiles.py", HERE / "momentum_directions.py",
             HERE / "compare_momentum.py", HERE / "compare_objectives.py",
             HERE / "probes/capture.py", HERE / "probes/gradients.py"]
    digest = hashlib.sha256((HERE / "momentum_direction_results.json").read_bytes()).hexdigest()
    if digest != MOMENTUM_SHA256:
        raise RuntimeError("Frozen momentum observations changed")
    return {"lab": code(), "reader": {str(path.relative_to(HERE)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "torch": torch.__version__, "device": "cpu", "threads": 4,
            "lean_source_commit": "cb38c97", "momentum_results_sha256": digest,
            "training_resumed": False}


def parity(measurement, frozen):
    """Check independent native endpoints and retain every slope discrepancy."""
    if measurement["rate"] != frozen["rate"] or any(
        measurement["minibatch"][key] != frozen["minibatch"][key]
        for key in ("examples", "place", "indices_sha256")
    ):
        raise RuntimeError("Disposable native rate or next minibatch differs")
    result = {}
    for name, kind in (("train_all", "full"), ("heldout_nonzero", "answer")):
        records = measurement["populations"][name]["records"]
        old = frozen["populations"][name]
        differences = {}
        for label, observed in (("before", records[0]), ("after", records[-1])):
            expected = old[label]
            if observed["examples"] != expected["examples"] or observed["answer_accuracy"] != expected["answer_accuracy"]:
                raise RuntimeError("Native endpoint population or accuracy differs")
            expected_loss = expected["losses"][kind]
            difference = observed["loss"] - expected_loss
            if not math.isclose(observed["loss"], expected_loss, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError("Native endpoint CE differs from the frozen observer")
            differences[label + "_CE_difference"] = difference
        expected_slope = old["alignment"][kind]["finite_CPU_displacement"]["loss_rate_derivative"]
        observed_slope = records[0]["rate_derivative"]
        if not math.isclose(observed_slope, expected_slope, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError("Initial autograd slope differs materially from the frozen observer")
        differences.update(initial_slope_difference=observed_slope - expected_slope,
                           frozen_initial_slope=expected_slope)
        result[name] = differences
    return result


def main():
    torch.set_num_threads(4)
    provenance = identity()
    momentum = json.loads((HERE / "momentum_direction_results.json").read_text())
    if momentum["observer"] != momentum_identity():
        raise RuntimeError("Frozen momentum data do not match their source identity")
    reference = {(run["study"], run["label"], record["step"]): record
                 for run in momentum["runs"] for record in run["records"]}
    path = HERE / "curvature_profile_results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["observer"] != provenance:
        raise RuntimeError("Reader changed: preserve existing results under their original version")
    cached = {(run["study"], run["label"], record["step"]): record
              for run in previous.get("runs", []) for record in run["records"]}
    result = {"observer": provenance, "selection": "same_19_noninitial_native_moment_and_sampler_states",
              "interpretation": "sampled_HVP_is_not_an_interval_certificate; CPU_steps_are_counterfactual",
              "runs": [], "missing": [], "excluded": []}
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
            key = (study.name, label, step)
            old = reference[key]
            if digest != old["checkpoint_sha256"]:
                raise RuntimeError(f"Frozen momentum checkpoint changed: {checkpoint}")
            record = cached.get(key)
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
                          "measurement": measurement, "frozen_momentum_parity": parity(measurement, old["measurement"]),
                          "observation_seconds": time.monotonic() - started}
                print(study.name, label, step,
                      {name: {field: pop["records"][-1][field] for field in
                              ("loss_change", "quadratic_prediction", "quadratic_remainder")}
                       for name, pop in measurement["populations"].items()}, flush=True)
            run["records"].append(record)
            if identity() != provenance:
                raise RuntimeError("Observation source changed during measurement")
            replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    observed = {(run["study"], run["label"], record["step"])
                for run in result["runs"] for record in run["records"]}
    if observed != set(reference) or result["missing"] != momentum["missing"] or result["excluded"] != momentum["excluded"]:
        raise RuntimeError("Checkpoint coverage differs from the completed native-state reader")
    replace(path, json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
