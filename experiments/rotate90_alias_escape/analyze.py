"""Replay averaged checkpoints and test the full-table gradient prediction.

Source: review/flat_spectrum.md, Alias symmetry and training. These diagnostics
reuse the actual model, loss, task split and Schedule-Free evaluation transform.
The float64 gradient check explicitly retains the original float32 RMS epsilon;
casting a PyTorch model alone would otherwise change that default constant.

From python/: uv run --locked python ../experiments/rotate90_alias_escape/analyze.py
Optional positional mode: gradients, escape, or budget. No training occurs here.
"""

import csv
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys

import torch
from torch.nn import functional as F

from lab.dsl import PowerDivision
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.peri_restart import alias_projection
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.rotary_geometry import multiplicative_ordering
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "rotate90_power_learning"
SOURCES = runpy.run_path(str(ROOT / "experiment.py"))["sources"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def residual(tensor, size):
    value = tensor.detach().cpu().double()
    distance = (value - alias_projection(value, size)).norm()
    norm = value.norm()
    return {"norm": float(norm), "transverse_norm": float(distance),
            "relative": float(distance / norm) if norm else 0.}


def quotient(prime, size):
    generator, _ = multiplicative_ordering(prime)
    subgroup = [pow(generator, (prime - 1) // size * j, prime) for j in range(size)]
    return torch.tensor([min(q * h % prime for h in subgroup) for q in range(prime)])


def gradients():
    findings = []
    specs = load(SOURCE).experiments
    for prime, exponent, size, seed, label, expected in SOURCES:
        if seed != 11:
            continue
        checkpoint = SOURCE / "runs" / label / "checkpoint.pt"
        assert digest(checkpoint) == expected
        spec = specs[label]
        model = build_model(spec.model, prime, seed).double()
        model.load_state_dict(RunDirectory(checkpoint.parent).checkpoint("cpu")["model"])
        model.normalize = lambda x: F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)
        initial = residual(model.E, size)
        with torch.no_grad():
            model.E.copy_(alias_projection(model.E, size))
        for power in (exponent, size):
            for fraction in (1., .5):
                task = build_task(PowerDivision(prime, power, fraction), seed, torch.device("cpu"))
                rows = task.splits["train"]
                model.zero_grad(set_to_none=True)
                loss_sum = 0.
                for batch in rows.split(2048):
                    loss = F.cross_entropy(model(batch[:, :2]), batch[:, 2], reduction="sum") / len(rows)
                    loss.backward()
                    loss_sum += float(loss.detach())
                value = {"source": label, "checkpoint_sha256": expected, "prime": prime,
                         "exponent": power, "subgroup_order": size, "train_fraction": fraction,
                         "initial_embedding": initial, "loss": loss_sum,
                         "embedding_gradient": residual(model.E.grad, size),
                         "task": task.metadata}
                findings.append(value)
                print("gradient", prime, power, fraction, value["embedding_gradient"]["relative"], flush=True)
    write_json(ROOT / "gradient_checks.json", {
        "script_sha256": digest(Path(__file__)), "python": platform.python_version(), "torch": torch.__version__,
        "arithmetic": "CPU float64, actual forward with explicit original float32 RMSNorm epsilon",
        "epsilon": torch.finfo(torch.float32).eps,
        "prediction": "Only full-table rows with coprime exponent/subgroup order must have zero transverse gradient",
        "checks": findings})


@torch.no_grad()
def replay(spec, directory, size):
    checkpoint = directory / "checkpoint.pt"
    state = RunDirectory(directory).checkpoint("cpu")
    model = build_model(spec.model, spec.benchmark.prime, spec.seeds.model)
    model.load_state_dict(state["model"])
    optimizer = build_optimizer(spec.optimizer, model)
    optimizer.load_state_dict(state["optimizer"])
    assert optimizer.param_groups[0]["training"] is True
    query = residual(model.E, size)
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    mapping = quotient(spec.benchmark.prime, size)
    measured = {}
    with evaluation_weights(optimizer):
        averaged = residual(model.E, size)
        for split, rows in task.splits.items():
            if not len(rows):
                continue
            correct = quotient_correct = 0
            loss = 0.
            for chunk in rows.split(1024):
                logits = model(chunk[:, :2])
                predictions = logits.argmax(-1)
                correct += int((predictions == chunk[:, 2]).sum())
                quotient_correct += int((mapping[predictions] == mapping[chunk[:, 2]]).sum())
                loss += float(F.cross_entropy(logits, chunk[:, 2], reduction="sum"))
            measured[split] = {"loss": loss / len(rows), "accuracy": correct / len(rows),
                               "quotient_accuracy": quotient_correct / len(rows), "examples": len(rows)}
    return {"checkpoint_sha256": digest(checkpoint), "query_embedding": query,
            "averaged_embedding": averaged, "cpu_float32_replay": measured}


def summarize(folder):
    study = load(folder)
    results, curves = [], []
    common_provenance = None
    for label, spec in study.experiments.items():
        directory = folder / "runs" / label
        result = json.loads((directory / "result.json").read_text())
        manifest = json.loads((directory / "experiment.json").read_text())
        assert result["updates"] == spec.budget.updates
        assert manifest["segments"][-1]["source_sha256"] == digest(folder / "experiment.py")
        provenance = manifest["segments"][-1]["provenance"]
        if common_provenance is None:
            common_provenance = provenance
        assert provenance == common_provenance
        size = 3 if spec.benchmark.prime == 97 else 8
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        summary = {"label": label, "prime": spec.benchmark.prime, "exponent": spec.benchmark.exponent,
                   "seed": spec.seeds.model, "width": spec.model.width, "depth": spec.model.depth,
                   "decay": spec.optimizer.weight_decay, "subgroup_order_for_diagnostic": size,
                   "updates": result["updates"], "training_seconds": result["training_seconds"],
                   "epochs_seen": result["final"]["epochs_seen"], "parameters": result["final"]["parameters"],
                   "train": result["final"]["train"], "heldout": result["final"].get("heldout"),
                   "best_heldout_accuracy": result.get("best_heldout_accuracy"),
                   "first_heldout99_update": result.get("first_heldout99_update"),
                   "at_5000": next((h for h in history if h["step"] == 5000), None),
                   "task": result["final"]["task"], **replay(spec, directory, size)}
        summary["success99_final"] = bool(summary["heldout"] and summary["heldout"]["accuracy"] >= .99)
        summary["replay_accuracy_delta"] = {
            split: metrics["accuracy"] - result["final"][split]["accuracy"]
            for split, metrics in summary["cpu_float32_replay"].items()}
        results.append(summary)
        for row in history:
            for split in ("train", "heldout"):
                if split in row and row[split]:
                    curves.append({"label": label, "step": row["step"], "split": split,
                                   "loss": row[split]["loss"], "accuracy": row[split]["accuracy"]})
        print(folder.name, label, summary["success99_final"], summary["heldout"], flush=True)
    write_json(folder / "results.json", {
        "analysis_script_sha256": digest(Path(__file__)),
        "definitions_sha256": digest(folder / "experiment.py"),
        "hypotheses_sha256": digest(folder / "hypotheses.json"), "training_provenance": common_provenance,
        "replay": "CPU FP32 eager, query restored then source optimizer eval() installs averaged weights",
        "metrics": "Primary accuracy/loss are recorded compiled-CUDA evaluation averages. CPU replay drift is retained separately. Quotient accuracy merges the specified multiplicative subgroup; it is not task accuracy.",
        "success_count": sum(row["success99_final"] for row in results), "runs": results})
    with (folder / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "step", "split", "loss", "accuracy"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    with (folder / "summary.csv").open("w", newline="") as stream:
        fields = ["label", "prime", "exponent", "seed", "width", "decay", "updates", "epochs_seen",
                  "train_accuracy", "heldout_accuracy", "best_heldout_accuracy", "success99_final",
                  "embedding_residual", "heldout_quotient_accuracy"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            flat = {key: row[key] for key in fields if key in row}
            flat.update(train_accuracy=row["train"]["accuracy"],
                        heldout_accuracy=row["heldout"]["accuracy"] if row["heldout"] else None,
                        embedding_residual=row["averaged_embedding"]["relative"],
                        heldout_quotient_accuracy=row["cpu_float32_replay"].get("heldout", {}).get("quotient_accuracy"))
            writer.writerow(flat)


def sources():
    specs = load(SOURCE).experiments
    rows = []
    for prime, exponent, size, seed, label, expected in SOURCES:
        directory = SOURCE / "runs" / label
        result = json.loads((directory / "result.json").read_text())
        assert digest(directory / "checkpoint.pt") == expected
        rows.append({"label": label, "prime": prime, "exponent": exponent, "seed": seed,
                     "original_updates": result["updates"], "train": result["final"]["train"],
                     "heldout": result["final"]["heldout"], **replay(specs[label], directory, size)})
    write_json(ROOT / "sources.json", rows)


def main():
    torch.set_num_threads(1)
    mode = sys.argv[1] if len(sys.argv) == 2 else "all"
    if mode not in {"all", "gradients", "escape", "budget"}:
        raise SystemExit("mode must be gradients, escape, budget, or all")
    if mode in {"all", "gradients"}:
        gradients()
    if mode in {"all", "escape"}:
        sources()
        summarize(ROOT)
    if mode in {"all", "budget"}:
        summarize(ROOT.parent / "rotate90_alias_budget")


if __name__ == "__main__":
    main()
