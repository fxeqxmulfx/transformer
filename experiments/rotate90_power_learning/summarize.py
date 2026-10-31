"""Export measured run records without changing or selecting checkpoints.

Run from python/: uv run --locked python ../experiments/rotate90_power_learning/summarize.py
The JSON and CSV are portable; ignored raw checkpoints remain under runs/.
"""

import csv
from collections import defaultdict
import hashlib
import json
import statistics
from pathlib import Path

from lab.infrastructure.loader import load

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def collect():
    rows, curves, incomplete = [], [], []
    for folder in sorted((ROOT / "runs").iterdir()):
        if not folder.is_dir() or not (folder / "experiment.json").exists():
            continue
        if not (folder / "result.json").exists():
            incomplete.append(folder.name)
            continue
        manifest, result = read(folder / "experiment.json"), read(folder / "result.json")
        config = manifest["description"]
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        final = result["final"]
        provenance = manifest["segments"][-1]["provenance"]
        heldout = final.get("heldout")
        late = [point["heldout"]["accuracy"] for point in history
                if point["step"] >= 3500 and "heldout" in point]
        row = {"label": folder.name, "prime": config["benchmark"]["prime"],
               "exponent": config["benchmark"]["exponent"], "width": config["model"]["width"],
               "depth": config["model"]["depth"], "parameters": final["parameters"],
               "optimizer": config["optimizer"]["type"],
               "optimizer_warmup": config["optimizer"].get("warmup", 0),
               "schedule_warmup": config["schedule"]["warmup"],
               "lr": config["optimizer"]["lr"], "weight_decay": config["optimizer"]["weight_decay"],
               "beta1": config["optimizer"]["betas"][0], "beta2": config["optimizer"]["betas"][1],
               "seed": config["seeds"]["model"], "data_seed": config["seeds"]["data"],
               "train_fraction": config["benchmark"]["train_fraction"],
               "updates": result["stop"]["step"], "train_accuracy": final["train"]["accuracy"],
               "heldout_accuracy": None if heldout is None else heldout["accuracy"],
               "train_loss": final["train"]["loss"],
               "heldout_loss": None if heldout is None else heldout["loss"],
               "best_heldout_accuracy": result["best_heldout_accuracy"],
               "late_heldout_accuracy_std": statistics.pstdev(late) if late else None,
               "best_minus_final_heldout_accuracy": (None if heldout is None else
                    result["best_heldout_accuracy"] - heldout["accuracy"]),
               "first_train99_update": result["first_train99_update"],
               "first_heldout99_update": result["first_heldout99_update"],
               "label_classes": final["task"]["label_classes"],
               "heldout_majority_accuracy": final["task"]["heldout_majority_accuracy"],
               "training_seconds": result["training_seconds"],
               "setup_seconds_including_compilation": history[0]["wall_seconds"],
               "wall_seconds": result["wall_seconds"],
               "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
               "python": provenance["python"], "torch": provenance["engine"]["torch"],
               "cuda": provenance["cuda"], "device": provenance["engine"]["device"],
               "engine_sha256": hashlib.sha256(json.dumps(provenance["engine"]["code"],
                                                          sort_keys=True).encode()).hexdigest(),
               "operand_train_fingerprint": final["task"]["operand_train_fingerprint"],
               "operand_heldout_fingerprint": final["task"]["operand_heldout_fingerprint"],
               "checkpoint_sha256": hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()}
        rows.append(row)
        for point in history:
            curves.append({"label": folder.name, "step": point["step"],
                           "train_accuracy": point["train"]["accuracy"],
                           "heldout_accuracy": point.get("heldout", {}).get("accuracy"),
                           "train_loss": point["train"]["loss"],
                           "heldout_loss": point.get("heldout", {}).get("loss")})
    return rows, curves, incomplete


def main():
    rows, curves, incomplete = collect()
    declared = set(load(ROOT).experiments)
    completed = {row["label"] for row in rows}
    not_started = sorted(declared - completed - set(incomplete))
    grouped = defaultdict(list)
    for row in rows:
        if row["optimizer"] == "ScheduleFreeAdamW":
            key = tuple(row[field] for field in ("prime", "exponent", "width", "depth", "lr",
                                                 "weight_decay", "beta1", "beta2"))
            grouped[key].append(row)
    groups = []
    for key, cases in sorted(grouped.items()):
        accuracies = [case["heldout_accuracy"] for case in cases]
        groups.append({**dict(zip(("prime", "exponent", "width", "depth", "lr", "weight_decay",
                                  "beta1", "beta2"), key, strict=True)),
                       "runs": [case["label"] for case in cases],
                       "seeds": [case["seed"] for case in cases],
                       "heldout_mean": statistics.mean(accuracies),
                       "heldout_std": statistics.pstdev(accuracies),
                       "heldout_min": min(accuracies), "heldout_max": max(accuracies),
                       "max_late_heldout_std": max(case["late_heldout_accuracy_std"] for case in cases),
                       "max_best_minus_final": max(case["best_minus_final_heldout_accuracy"] for case in cases)})
    payload = {"scope": "exploratory finite-budget training; heldout is used for sweep comparisons",
               "declared": len(declared), "completed": len(rows), "incomplete": incomplete,
               "not_started": not_started, "unexpected": sorted(completed - declared),
               "hypotheses_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in ("hypotheses.json", "large_model_hypotheses.json")},
               "schedule_free_groups": groups, "runs": rows}
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for name, table in (("summary.csv", rows), ("curves.csv", curves)):
        if table:
            with (ROOT / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    print(json.dumps({"declared": len(declared), "completed": len(rows), "incomplete": incomplete,
                      "not_started": not_started,
                      "heldout_99": sum(row["heldout_accuracy"] is not None and row["heldout_accuracy"] >= .99
                                        for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
