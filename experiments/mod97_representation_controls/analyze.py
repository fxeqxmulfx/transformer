"""Audit and export all prespecified M9.3b MLP controls, with both error units.

The fixed experiment grid is read in full. No held-out checkpoint selection
or seed filtering is performed. Run from python/ with uv run --locked python
../experiments/mod97_representation_controls/analyze.py.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from lab.domain.paired import VectorReadout
from lab.domain.spec import describe, fingerprint
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectories

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def first99(history, split):
    return next((row["step"] for row in history if row[split]["accuracy"] >= .99), None)


def main():
    torch.set_num_threads(1)
    study, runs = load(HERE), RunDirectories(HERE / "runs")
    assert len(study.experiments) == 36, "The prespecified grid changed"
    current_code, engine, splits, summaries, curves = code(), None, {}, {}, []
    for label, spec in study.experiments.items():
        run, budget = runs.open(label), spec.budget.updates
        result, manifest = run.result(), run.manifest()
        assert result is not None and result["updates"] == budget, f"Incomplete run: {label}"
        assert result["stop"]["reason"] == "budget"
        assert manifest["description"] == describe(spec), f"Changed configuration: {label}"
        provenance = run.provenance()
        recorded = provenance["engine"]
        assert recorded["code"] == current_code, "Restore the recorded lab sources before replay"
        assert recorded["torch"] == torch.__version__, "Restore the recorded PyTorch runtime"
        assert provenance["commit"]["lab_modified"] is False, "Lab sources were not committed"
        engine = recorded if engine is None else engine
        assert engine == recorded, "The controls used different engines"
        history, gradients, diagnostics = (run.records(name) for name in ("history", "gradients", "diagnostics"))
        assert [row["step"] for row in history] == list(range(0, budget + 1, spec.evaluate.every))
        assert [row["step"] for row in gradients] == list(range(1, budget + 1))
        assert [row["step"] for row in diagnostics] == list(range(spec.diagnostics.every, budget + 1,
                                                                 spec.diagnostics.every))
        assert finite(history) and finite(gradients) and finite(diagnostics) and finite(result)
        device = torch.device(spec.execution.device)
        task = build_task(spec.benchmark, spec.seeds.data, device, spec.model)
        rows = torch.cat(tuple(task.splits.values()))
        assert len(rows) == 9312 and all(len(part) == 4656 for part in task.splits.values())
        assert len(torch.unique(rows[:, 0] * 97 + rows[:, 1])) == 9312
        assert bool(((rows[:, 1] > 0) & (rows[:, 1] < 97) &
                     (rows[:, 0] >= 0) & (rows[:, 0] < 97) &
                     (rows[:, 2] >= 0) & (rows[:, 2] < 97) &
                     ((rows[:, 1] * rows[:, 2] - rows[:, 0]) % 97 == 0)).all())
        split_hashes = {name: tensor_hash(part) for name, part in task.splits.items()}
        assert splits.setdefault(spec.seeds.data, split_hashes) == split_hashes
        model = build_model(spec.model, task.vocab, spec.seeds.model).to(device)
        initial_hashes = {name: tensor_hash(value) for name, value in model.state_dict().items()}
        checkpoint = run.checkpoint(device)
        assert checkpoint["step"] == budget
        model.load_state_dict(checkpoint["model"])
        names = set(dict(model.named_parameters()))
        for row in diagnostics:
            assert set(row["parameters"]) == names
            for fields in row["parameters"].values():
                assert {"update_l2", "exp_avg_l2", "exp_avg_sq_l2"} <= fields.keys()
        for split in ("train", "heldout"):
            assert evaluate(task, model, task.splits[split], spec.evaluate.batch) == result["final"][split]
        crossings = {split: first99(history, split) for split in ("train", "heldout")}
        for split, first in crossings.items():
            assert result[f"first_{split}99_update"] == first
        scalar = not isinstance(spec.model, VectorReadout)
        for row in history:
            if scalar:
                for split in ("train", "heldout"):
                    metrics = row[split]
                    assert math.isclose(metrics["continuous_numeric_mse"], 4608 * metrics["loss"],
                                        rel_tol=2e-5, abs_tol=2e-8), "Scalar MSE units changed"
            curves.append({"label": label, "update": row["step"],
                           **{f"{split}_{metric}": row[split][metric]
                              for split in ("train", "heldout")
                              for metric in ("loss", "accuracy", "numeric_mse")},
                           **{f"{split}_continuous_numeric_mse": row[split].get("continuous_numeric_mse", "")
                              for split in ("train", "heldout")},
                           **{key: row[key] for key in ("parameter_norm", "first_preactivation_rms",
                                                        "first_activation_rms")}})
        summaries[label] = {
            "experiment_fingerprint": fingerprint(spec), "split": task.summary(),
            "split_sha256": split_hashes, "updates": budget,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_parameter_sha256": initial_hashes,
            **{f"first_{split}99_update": first for split, first in crossings.items()},
            "maximum_train_accuracy": max(row["train"]["accuracy"] for row in history),
            "maximum_heldout_accuracy": max(row["heldout"]["accuracy"] for row in history),
            "final": result["final"], "initial": history[0],
            "common_10000_update_observation": next(row for row in history if row["step"] == 10000),
            "training_seconds": result["training_seconds"], "wall_seconds": result["wall_seconds"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "final_gradient_l2": gradients[-1]["gradient_l2"],
            "final_max_coordinate_abs": max(float(p.detach().abs().max()) for p in model.parameters()),
            "trajectory_start_commit": provenance["commit"],
            **{name + "_sha256": sha256(run.path / (name + suffix))
               for name, suffix in (("history", ".jsonl"), ("gradients", ".jsonl"),
                                    ("diagnostics", ".jsonl"), ("checkpoint", ".pt"))},
        }
        print(label, "final train/heldout", result["final"]["train"]["accuracy"],
              result["final"]["heldout"]["accuracy"], "first99", crossings, flush=True)
    for prefix in ("scalar", "onehot"):
        for decay in ("wd0", "wd1"):
            for seed in ("s11", "s17", "s23"):
                names = [f"{prefix}-{head}-{decay}-{seed}" for head in ("number-mse", "vector-mse", "vector-ce")]
                hashes = [summaries[name]["initial_parameter_sha256"] for name in names]
                assert all(hashes[0][key] == hashes[1][key] == hashes[2][key]
                           for key in hashes[0] if key.startswith("hidden."))
                assert hashes[1] == hashes[2], "The vector loss intervention changed initialization"
    portable = {"source_revision": "f49969393a7482490d45483d8a4fd3046b1adede",
                "analysis_script_sha256": sha256(Path(__file__)),
                "engine": {key: value for key, value in engine.items() if key != "code"},
                "cuda": provenance["cuda"], "python": provenance["python"],
                "engine_code_sha256": hashlib.sha256(json.dumps(current_code, sort_keys=True).encode()).hexdigest(),
                "runs": summaries}
    (HERE / "results.json").write_text(json.dumps(portable, indent=2, allow_nan=False) + "\n")
    (HERE / "engine_code.json").write_text(json.dumps(current_code, indent=2, sort_keys=True) + "\n")
    with (HERE / "curves.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)


if __name__ == "__main__":
    main()
