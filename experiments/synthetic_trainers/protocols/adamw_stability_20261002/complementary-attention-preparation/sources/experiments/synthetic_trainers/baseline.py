"""Measure a reproducible AMSGradW/softmax baseline on the synthetic suite."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch

from .baseline_recipes import recipes
from .runtime import write_json
from .training import train_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "suite", "transitions", "capacity", "control"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--suite-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=5000)
    parser.add_argument("--capacity-steps", type=int, default=1000)
    parser.add_argument("--control-steps", type=int, default=1000)
    args = parser.parse_args(argv)
    plan = recipes(args.phase, device=args.device, seeds=tuple(args.seeds),
                   suite_steps=args.suite_steps, transition_steps=args.transition_steps,
                   capacity_steps=args.capacity_steps, control_steps=args.control_steps)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "planned_runs": len(plan), "completed_runs": 0,
                "attention": "softmax", "dtype": "float32", "torch": torch.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "device": args.device, "cpu_threads": torch.get_num_threads(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                "gpu_memory_bytes": torch.cuda.get_device_properties(args.device).total_memory if args.device.startswith("cuda") else None,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "recipes": [{"name": recipe.name, **asdict(recipe)} for recipe in plan],
                "interpretation": "calibration_and_finite_curve_candidates; fixed_data_seed; no_causal_claim"}
    write_json(args.output / "manifest.json", manifest)
    completed = []
    started = time.perf_counter()
    for i, recipe in enumerate(plan, 1):
        print(json.dumps({"event": "start", "run": recipe.name, "index": i, "total": len(plan)}), flush=True)

        def progress(row):
            novel = row["validation_novel"]
            print(json.dumps({"event": "observe", "run": recipe.name, "step": row["step"],
                              "train_sequence_accuracy": row["train"]["sequence_accuracy"],
                              "validation_novel_accuracy": novel["sequence_accuracy"] if novel else None,
                              "validation_loss": row["validation"]["example_loss"],
                              "ood_accuracy": {name: score["sequence_accuracy"] for name, score in row["validation_ood"].items()},
                              "training_seconds": row["training_seconds"]}), flush=True)

        result = train_run(recipe.task, recipe.model, recipe.training, args.output / recipe.name, progress=progress)
        completed.append({"name": recipe.name, "result": f"{recipe.name}/result.json"})
        manifest["completed_runs"] = len(completed)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "runs.json", completed)
        print(json.dumps({"event": "complete", "run": recipe.name, "total_wall_seconds": result["total_wall_seconds"],
                          "test_id_accuracy": result["test_final"]["in_distribution"]["sequence_accuracy"],
                          "transition": result["study"]["generalization_transition"]}), flush=True)
    manifest["status"] = "complete"
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
