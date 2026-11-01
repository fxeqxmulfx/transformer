"""Freeze, run, then replay every task155 case; source edits invalidate continuation.

Run from python/: uv run --locked python ../experiments/rms_backward/cohort.py freeze|run|report.
The normal lab DSL/Engine performs every update. Test is opened only in report
after all cases have an endpoint (including any explicitly retained failure).
"""

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

from experiment import experiments
from lab.application.study import Study, run_study
from lab.dsl import describe, fingerprint
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine import Engine
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectories, replace

HERE = Path(__file__).resolve().parent


def write(path, value):
    replace(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}


def freeze():
    path = HERE / "protocol.json"
    if path.exists():
        raise FileExistsError("The frozen cohort cannot be replaced")
    write(path, {"python": platform.python_version(), "torch": str(torch.__version__),
                 "engine": code(), "sources": sources(), "cpu_only": True,
                 "cases": {name: {"fingerprint": fingerprint(spec), "description": describe(spec)}
                           for name, spec in experiments.items()}})


def check():
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert protocol["engine"] == code(), "Frozen engine changed"
    assert protocol["sources"] == sources(), "Frozen orchestration changed"
    assert protocol["python"] == platform.python_version()
    assert protocol["torch"] == str(torch.__version__)
    assert list(protocol["cases"]) == list(experiments)
    for name, spec in experiments.items():
        assert protocol["cases"][name]["fingerprint"] == fingerprint(spec)
        assert spec.execution.device == "cpu"
    return protocol


def run():
    check()
    ledger = HERE / "runs/cohort.json"
    data = json.loads(ledger.read_text()) if ledger.exists() else {"closed": False, "attempts": []}
    if data["closed"]:
        raise RuntimeError("Learning is closed; no post-test extension")
    source = (HERE / "experiment.py").read_text()
    study = Study(HERE.name, source, experiments)
    runs = RunDirectories(HERE / "runs")
    for label in experiments:
        if any(a["label"] == label for a in data["attempts"]):
            continue
        started = time.perf_counter()
        try:
            result = run_study(study, [label], runs, Engine(), lambda name, row: None)[label]
            event = {"label": label, "status": "complete", "stop": result["stop"]}
        except Exception:
            event = {"label": label, "status": "failed", "traceback": traceback.format_exc()}
        event["wall_seconds"] = time.perf_counter() - started
        data["attempts"].append(event)
        write(ledger, data)
        print(json.dumps(event), flush=True)
    data["closed"] = True
    write(ledger, data)


def report():
    protocol = check()
    ledger = json.loads((HERE / "runs/cohort.json").read_text())
    assert ledger["closed"] and len(ledger["attempts"]) == len(experiments)
    torch.set_num_threads(1)
    rows = []
    for event in ledger["attempts"]:
        label = event["label"]
        spec = experiments[label]
        run_dir = RunDirectories(HERE / "runs").open(label)
        row = {**event, "history": run_dir.records("history")}
        checkpoint = run_dir.checkpoint("cpu")
        if checkpoint is not None:
            task = build_task(spec.benchmark, spec.seeds.data, "cpu", spec.model)
            model = build_model(spec.model, 0, spec.seeds.model)
            optimizer = build_optimizer(spec.optimizer, model)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            with evaluation_weights(optimizer):
                row["final"] = {name: evaluate(task, model, split, 256) for name, split in task.splits.items()}
                coefficients = model.coefficients().detach().cpu()
                errors = (coefficients - task.teacher).abs()
                row["coefficients"] = coefficients.tolist()
                row["coefficient_max_error"] = errors.max().item()
                row["uniform_l2_bound"] = spec.model.width**.5 * errors.sum().item()
            row["split_sha256"] = task.hashes
            gradients = [x["gradient_l2"] for x in run_dir.records("gradients")]
            row["gradients"] = {"count": len(gradients), "first": gradients[0] if gradients else None,
                                "last": gradients[-1] if gradients else None,
                                "zero_count": sum(x == 0 for x in gradients),
                                "minimum": min(gradients) if gradients else None,
                                "maximum": max(gradients) if gradients else None}
            row["checkpoint_sha256"] = hashlib.sha256((run_dir.path / "checkpoint.pt").read_bytes()).hexdigest()
            row["recovered"] = (event["status"] == "complete" and row["final"]["test"]["relative_mse"] <= 1e-4
                                and row["coefficient_max_error"] <= .01)
        rows.append(row)
        print(label, row.get("recovered"), row.get("coefficient_max_error"), flush=True)
    write(HERE / "results.json", {"protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                                 "ledger": ledger, "cases": rows})


if __name__ == "__main__":
    {"freeze": freeze, "run": run, "report": report}[sys.argv[1]]()
