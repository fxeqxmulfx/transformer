"""Export every matched run and evaluate predetermined embedding projections.

Source: landscape_grokking_plan.md, M9.3--M9.4. Character pairs are selected
by final embedding energy, without held-out labels. The common row mean,
zero row and all branch matrices are retained. Replacing E changes all three
of its tied roles. This is a checkpoint ablation, not a training intervention.
An optional experiment-folder argument selects another modulus-97 study.
Run from python/: uv run --locked python ../experiments/rotate90_analytic/analyze.py
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from lab.domain.spec import describe, fingerprint
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.rotary_geometry import multiplicative_ordering
from lab.infrastructure.store import RunDirectories

HERE = Path(__file__).resolve().parent
COUNTS = (0, 1, 2, 4, 8, 16, 48)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def character_keys(geometry, fraction=0.9):
    ordered = sorted(enumerate(geometry["character_pair_shares"], 1), key=lambda pair: -pair[1])
    selected, energy = [], 0.0
    for frequency, share in ordered:
        selected.append(frequency)
        energy += share
        if energy >= fraction:
            break
    return selected


@torch.no_grad()
def projections(model, task, batch):
    original = model.E.detach().clone()
    _, residues = multiplicative_ordering(task.vocab)
    order = torch.tensor(residues, device=original.device)
    rows = original[order].double()
    mean = rows.mean(dim=0)
    spectrum = torch.fft.fft(rows - mean, dim=0)
    geometry = task.observe(model, None)["embedding_geometry"]
    ordered = sorted(range(1, 49), key=lambda k: -geometry["character_pair_shares"][k - 1])
    output = []
    for count in COUNTS:
        frequencies = ordered[:count]
        mask = torch.zeros(len(rows), device=original.device, dtype=torch.bool)
        for frequency in frequencies:
            mask[frequency] = mask[-frequency] = True
        reconstructed = (torch.fft.ifft(spectrum * mask[:, None], dim=0).real + mean).to(original.dtype)
        model.E.copy_(original)
        model.E[order] = reconstructed
        assert torch.equal(model.E[0], original[0]), "The zero-class row changed"
        if count == 48:
            assert torch.equal(model.E, original), "Full character reconstruction changed FP32 parameters"
        output.append({"pairs": count, "frequencies": frequencies,
                       "retained_centered_energy": sum(geometry["character_pair_shares"][k - 1]
                                                       for k in frequencies),
                       "joint_character_order": 96 // math.gcd(96, *frequencies),
                       **{split: evaluate(task, model, task.splits[split], batch)
                          for split in ("train", "heldout")}})
    model.E.copy_(original)
    return output


def main(folder=HERE):
    torch.set_num_threads(1)
    folder = Path(folder).resolve()
    study, runs = load(folder), RunDirectories(folder / "runs")
    current_code, common_engine, summaries, curves = code(), None, {}, []
    for label, experiment in study.experiments.items():
        run = runs.open(label)
        result, manifest = run.result(), run.manifest()
        budget = experiment.budget.updates
        assert result is not None and result["updates"] == budget, f"Incomplete run: {label}"
        assert manifest["description"] == describe(experiment), f"Changed experiment: {label}"
        provenance = run.provenance()
        engine = provenance["engine"]
        assert engine["code"] == current_code, f"Restore the recorded lab code before analyzing {label}"
        assert engine["torch"] == torch.__version__, "Restore the recorded PyTorch runtime"
        common_engine = engine if common_engine is None else common_engine
        assert engine == common_engine, "The controls used different engines"
        history, gradients = run.records("history"), run.records("gradients")
        assert [row["step"] for row in history] == list(range(0, budget + 1, 10))
        assert [row["step"] for row in gradients] == list(range(1, budget + 1))
        assert all(math.isfinite(row["gradient_l2"]) for row in gradients)
        device = torch.device(experiment.execution.device)
        task = build_task(experiment.benchmark, experiment.seeds.data, device)
        assert task.vocab == 97, "The exported character pairs require modulus 97"
        model = build_model(experiment.model, task.vocab, experiment.seeds.model).to(device)
        checkpoint = run.checkpoint(device)
        assert checkpoint["step"] == budget
        model.load_state_dict(checkpoint["model"])
        for split in ("train", "heldout"):
            assert evaluate(task, model, task.splits[split], experiment.evaluate.batch) == result["final"][split]
        first = result["first_heldout99_update"]
        after_first = [row for row in history if first is not None and row["step"] >= first]
        last_bad = max((row["step"] for row in history if row["heldout"]["accuracy"] < .99), default=-10)
        sustained = next((row["step"] for row in history if row["step"] > last_bad), None)
        ablations = projections(model, task, experiment.evaluate.batch)
        assert ablations[-1]["train"] == result["final"]["train"]
        assert ablations[-1]["heldout"] == result["final"]["heldout"]
        summaries[label] = {
            "experiment_fingerprint": fingerprint(experiment), "split": task.summary(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "updates": result["updates"], "first_train99_update": result["first_train99_update"],
            "first_heldout99_update": first, "sustained_heldout99_from_update": sustained,
            "post99_observations_below99": sum(row["heldout"]["accuracy"] < .99 for row in after_first),
            "worst_post99_observation": min(after_first, key=lambda row: row["heldout"]["accuracy"], default=None),
            "last_1000_updates_min_heldout_accuracy": min(row["heldout"]["accuracy"] for row in history
                                                          if row["step"] >= budget - 1000),
            "nominal_decay_clock_at_first99": (None if first is None else experiment.optimizer.weight_decay *
                math.fsum(row["learning_rate"] for row in gradients if row["step"] <= first)),
            "nominal_decay_clock_at_end": experiment.optimizer.weight_decay *
                math.fsum(row["learning_rate"] for row in gradients),
            "training_seconds": result["training_seconds"], "wall_seconds": result["wall_seconds"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "initial_geometry": history[0]["embedding_geometry"],
            "first_heldout99_geometry": next((row["embedding_geometry"] for row in history
                                             if row["step"] == first), None),
            "final": result["final"], "final_gradient_l2": gradients[-1]["gradient_l2"],
            "final_max_coordinate_abs": max(float(parameter.detach().abs().max()) for parameter in model.parameters()),
            "final_character_keys90": character_keys(result["final"]["embedding_geometry"]),
            "trajectory_start_commit": provenance["commit"],
            "history_sha256": sha256(run.path / "history.jsonl"),
            "gradients_sha256": sha256(run.path / "gradients.jsonl"),
            "checkpoint_sha256": sha256(run.path / "checkpoint.pt"),
            "character_projections": ablations,
        }
        prefix = [row for row in history if row["step"] <= 10000]
        prefix_first = next((row["step"] for row in prefix if row["heldout"]["accuracy"] >= .99), None)
        prefix_after = [row for row in prefix if prefix_first is not None and row["step"] >= prefix_first]
        summaries[label]["common_10000_update_prefix"] = {
            "last_update": prefix[-1]["step"], "train": prefix[-1]["train"], "heldout": prefix[-1]["heldout"],
            "first_heldout99_update": prefix_first,
            "post99_observations_below99": sum(row["heldout"]["accuracy"] < .99 for row in prefix_after),
            "worst_post99_heldout_accuracy": min((row["heldout"]["accuracy"] for row in prefix_after), default=None),
        }
        for row in history:
            geometry = row["embedding_geometry"]
            keys = character_keys(geometry)
            curves.append({"label": label, "update": row["step"],
                           **{f"{split}_{metric}": row[split][metric]
                              for split in ("train", "heldout") for metric in ("loss", "accuracy")},
                           **{key: geometry[key] for key in ("effective_rank", "J_overlap",
                               "mean_energy_fraction", "largest_character_pair_share",
                               "dominant_character_frequency", "dominant_character_order")},
                           "character_pairs_for_90_percent": len(keys),
                           "joint_order_of_keys90": 96 // math.gcd(96, *keys),
                           **{key: row[key] for key in ("parameter_norm", "embedding_norm", "zero_embedding_norm")}})
        print(label, "first99", first, "sustained99", sustained,
              "projection heldout", [round(probe["heldout"]["accuracy"], 6) for probe in ablations], flush=True)
    portable = {"protocol_commit": "c59d4426", "source_revision": "f49969393a7482490d45483d8a4fd3046b1adede",
                "source_export_sha256": sha256(folder.parent.parent / "third_party/rotate90/reference.py"),
                "analysis_script_sha256": sha256(Path(__file__)),
                "engine": {key: value for key, value in common_engine.items() if key != "code"},
                "cuda": provenance["cuda"], "python": provenance["python"],
                "engine_code_sha256": hashlib.sha256(json.dumps(current_code, sort_keys=True).encode()).hexdigest(),
                "projection_pair_counts": COUNTS, "runs": summaries}
    (folder / "results.json").write_text(json.dumps(portable, indent=2, allow_nan=False) + "\n")
    (folder / "engine_code.json").write_text(json.dumps(current_code, indent=2, sort_keys=True) + "\n")
    with (folder / "curves.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: analyze.py [experiment_folder]")
    main(sys.argv[1] if len(sys.argv) == 2 else HERE)
