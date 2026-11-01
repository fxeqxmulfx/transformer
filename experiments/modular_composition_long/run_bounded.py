"""Run the frozen continuation cohort under one two-hour wall-clock cap.

Source: user-authorized budget, EXPERIMENT_PLAN.md (2026-11-01). Each child
uses the ordinary run.py trainer and its atomic full-state checkpoints.
Timeout stops the process group; the last complete checkpoint remains
resumable. No truncated run is labeled as a completed learning failure.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from lab.infrastructure.loader import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def run_child(command, cwd, log, deadline):
    """Leave five seconds for termination inside the shared deadline."""
    with log.open("w") as output:
        child = subprocess.Popen(command, cwd=cwd, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=max(0, deadline - time.monotonic() - 5)), False
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            return child.returncode, True


def main():
    study = load(HERE)
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    ledger = runs / "phase.json"
    if ledger.exists():
        raise FileExistsError("This phase has already started; no automatic second two-hour budget")
    for experiment in study.experiments.values():
        source = ROOT / experiment.checkpoint.source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == experiment.checkpoint.sha256
    started = time.monotonic()
    deadline = started + 7200
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "wall_budget_seconds": 7200,
              "protocol_sha256": hashlib.sha256((HERE / "hypotheses.json").read_bytes()).hexdigest(),
              "experiment_sha256": hashlib.sha256((HERE / "experiment.py").read_bytes()).hexdigest(),
              "cases": []}
    write(ledger, record)
    for label, experiment in study.experiments.items():
        case = {"label": label, "started_after_seconds": time.monotonic() - started,
                "status": "not_started"}
        record["cases"].append(case)
        if time.monotonic() + 5 < deadline:
            case["status"] = "running"
            write(ledger, record)
            print(json.dumps(case), flush=True)
            code, expired = run_child([str(ROOT / "run.py"), "run", str(HERE), label],
                                      ROOT, runs / f"{label}.log", deadline)
            result = runs / label / "result.json"
            completed = result.exists() and json.loads(result.read_text())["stop"] == {
                "step": experiment.budget.updates, "reason": "budget"}
            case.update(returncode=code, status="timeout" if expired else
                        "completed" if completed and code == 0 else "failed")
        case["ended_after_seconds"] = time.monotonic() - started
        write(ledger, record)
        print(json.dumps(case), flush=True)
    record["wall_seconds"] = time.monotonic() - started
    record["ended_utc"] = datetime.now(timezone.utc).isoformat()
    write(ledger, record)
    for experiment in study.experiments.values():
        source = ROOT / experiment.checkpoint.source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == experiment.checkpoint.sha256


if __name__ == "__main__":
    main()
