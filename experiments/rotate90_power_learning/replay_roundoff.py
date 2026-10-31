"""Resolve CPU/CUDA argmax differences by replaying the original compiled device.

Only checkpoints with discrepancies in audit.py are read. Preserve the strict CPU
comparison and certify differing decisions only when the original compiled CUDA
metrics match and the CPU/CUDA logits differ by float32 roundoff near a tie.
Run after audit.py, from python/: uv run --locked python ../experiments/rotate90_power_learning/replay_roundoff.py
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    audit = json.loads((ROOT / "checkpoint_audit.json").read_text())
    study = load(ROOT)
    reports = []
    for record in audit["runs"]:
        if all(row["accuracy_matches"] and row["loss_matches"] for row in record["splits"].values()):
            continue
        label = record["label"]
        experiment = study.experiments[label]
        run = RunDirectory(ROOT / "runs" / label)
        checkpoint = run.checkpoint("cpu")
        cpu_task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        cpu_model = build_model(experiment.model, cpu_task.vocab, experiment.seeds.model)
        cpu_model.load_state_dict(checkpoint["model"])
        cpu_optimizer = build_optimizer(experiment.optimizer, cpu_model)
        cpu_optimizer.load_state_dict(checkpoint["optimizer"])
        gpu_task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cuda"))
        gpu_model = build_model(experiment.model, gpu_task.vocab, experiment.seeds.model).cuda()
        gpu_model.load_state_dict(checkpoint["model"])
        stepper = CompiledStepper(experiment, gpu_task, gpu_model, None)
        stepper.load_state_dict(checkpoint["optimizer"])
        row = {"label": label, "checkpoint_sha256": hashlib.sha256(
            (ROOT / "runs" / label / "checkpoint.pt").read_bytes()).hexdigest(), "splits": {}}
        for split, comparison in record["splits"].items():
            if comparison["accuracy_matches"] and comparison["loss_matches"]:
                continue
            metrics = stepper.evaluate(split)
            expected = run.result()["final"][split]
            with evaluation_weights(cpu_optimizer), evaluation_weights(stepper.optimizer):
                cpu_model.eval()
                cpu_logits = torch.cat([cpu_model(chunk[:, :2]) for chunk in
                                        cpu_task.splits[split].split(experiment.evaluate.batch)])
                gpu_logits = torch.cat([stepper.inference(chunk[:, :2]).clone() for chunk in
                                        gpu_task.splits[split].split(experiment.evaluate.batch)]).cpu()
            cpu_pred, gpu_pred = cpu_logits.argmax(-1), gpu_logits.argmax(-1)
            changed = cpu_pred != gpu_pred
            difference = (cpu_logits - gpu_logits).abs()
            per_example_error = difference.max(-1).values
            cpu_top = cpu_logits.topk(2).values
            cpu_gap = cpu_top[:, 0] - cpu_top[:, 1]
            near_ties = bool((cpu_gap[changed] <= 2 * per_example_error[changed] + 1e-7).all())
            logits_close = bool(torch.allclose(cpu_logits, gpu_logits, rtol=5e-5, atol=5e-5))
            accuracy_matches = metrics["accuracy"] == expected["accuracy"]
            loss_matches = math.isclose(metrics["loss"], expected["loss"], rel_tol=5e-5, abs_tol=5e-6)
            item = {"compiled_cuda": metrics, "recorded": expected,
                    "accuracy_matches": accuracy_matches, "loss_matches": loss_matches,
                    "changed_decisions": int(changed.sum()), "all_changes_near_ties": near_ties,
                    "logits_close": logits_close, "max_logit_difference": float(difference.max()),
                    "changed_cpu_margins": cpu_gap[changed].tolist(),
                    "changed_example_max_errors": per_example_error[changed].tolist()}
            item["pass"] = accuracy_matches and loss_matches and near_ties and logits_close
            row["splits"][split] = item
        row["pass"] = all(item["pass"] for item in row["splits"].values())
        reports.append(row)
        print(json.dumps(row, indent=2), flush=True)
    (ROOT / "cuda_roundoff_replay.json").write_text(json.dumps({"runs": reports}, indent=2) + "\n")
    if not all(row["pass"] for row in reports):
        raise SystemExit("Some discrepancies remain unresolved")


if __name__ == "__main__":
    main()
