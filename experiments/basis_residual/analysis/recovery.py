"""Resume an interrupted Basis extension without changing its declared horizon.

The frozen engine restores model, optimizer, sampler and clocks. Preserve the
interrupted phase and all logs before it rewinds records beyond the checkpoint.
An interruption is not a budget exit or evidence against eventual learning.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

import torch

from continuation import (HERE, TERMINAL, BasisTraining, ResidualCompiledStepper,
                          RunDirectory, describe, frozen, load, provenance,
                          require_continuation, sha, swap, write)


def validate_recovery(phase, decision, directory, study):
    label = decision["label"]
    unfinished = {name for name, row in phase["cases"].items() if row["status"] not in TERMINAL}
    if unfinished != {label} or phase["status"] not in ("learning", "continuation_error", "interrupted"):
        raise ValueError("Recovery requires exactly one interrupted extension")
    record = phase["cases"][label]
    if record["status"] not in ("preparing", "learning", "continuation_error", "interrupted"):
        raise ValueError("This case is not an interrupted extension")
    previous = phase["continuations"][-1]
    if previous["decision_sha256"] != decision["continuation_sha256"]:
        raise ValueError("Recovery must name the latest continuation")
    plan = next(p for p in previous["decision"]["decisions"] if p["label"] == label)
    if plan["action"] != "extend" or decision["updates"] != plan["updates"]:
        raise ValueError("Recovery cannot change the declared update horizon")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError("Record why recovery is required")
    if sha(directory.path / "checkpoint.pt") != decision["checkpoint_sha256"]:
        raise ValueError("The recovery checkpoint changed")
    state = directory.checkpoint("cpu")
    if state["step"] != decision["step"] or not plan["step"] <= state["step"] < plan["updates"]:
        raise ValueError("Recovery must name a saved step inside the declared extension")
    spec = swap(study.experiments[label], "budget.updates", plan["updates"])
    require_continuation(directory.description(), spec)
    if directory.description()["budget"]["updates"] != plan["updates"]:
        raise ValueError("The interrupted manifest must already declare this horizon")
    history = [r for r in directory.records("history") if r["step"] <= state["step"]]
    if not history or history[-1]["step"] != state["step"] or any(spec.benchmark.stable([r]) for r in history):
        raise ValueError("Recovery requires an observed checkpoint with no earlier crossing")
    gradients = [r["step"] for r in directory.records("gradients") if r["step"] <= state["step"]]
    if gradients != list(range(1, state["step"] + 1)):
        raise ValueError("Every saved update must have its gradient record")
    return spec


def recover(decision_path):
    declaration = frozen()
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("Frozen runtime changed")
    if (HERE / "results.json").exists():
        raise ValueError("Learning cannot continue after final test has been opened")
    phase_path = HERE / "runs/phase.json"
    decision = json.loads(decision_path.read_text())
    if decision["protocol_sha256"] != sha(HERE / "protocol.json") or decision["phase_sha256"] != sha(phase_path):
        raise ValueError("The declared protocol or interrupted phase changed")
    phase = json.loads(phase_path.read_text())
    if set(phase["cases"]) != {case["label"] for case in declaration["cases"]}:
        raise ValueError("Every frozen case must remain present")
    label = decision["label"]
    directory, study = RunDirectory(HERE / "runs" / label), load(HERE)
    spec = validate_recovery(phase, decision, directory, study)
    archive = HERE / "runs/recoveries" / sha(decision_path)
    archive.mkdir(parents=True, exist_ok=False)
    # Preserve raw trailing records, even though the engine cannot resume them.
    for path in [phase_path, *directory.path.glob("*.json*"), directory.path / "checkpoint.pt"]:
        shutil.copyfile(path, archive / path.name)
    evidence = {"decision": decision, "runner_source_sha256": sha(Path(__file__)),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "preserved_files": {p.name: sha(p) for p in sorted(archive.iterdir())}}
    write(archive / "recovery.json", evidence)
    phase.setdefault("recoveries", []).append(evidence)
    record = phase["cases"][label]
    record["status"], phase["status"] = "preparing", "learning"
    write(phase_path, phase)
    start = time.monotonic()
    try:
        directory.begin(describe(spec), provenance(spec.execution.device), study.source)
        trainer = BasisTraining(spec, directory, lambda row: print(json.dumps({
            "label": label, "step": row["step"],
            "validation": row[spec.benchmark.validation_name]["sequence_accuracy"]}), flush=True),
            ResidualCompiledStepper)
        expected = next(case for case in declaration["cases"] if case["label"] == label)
        if trainer.task.fingerprints != expected["data"]["fingerprints"]:
            raise ValueError("Frozen Basis data changed")
        record["status"] = "learning"
        write(phase_path, phase)
        result = trainer()
        record.update(status=result["stop"]["reason"], stop=result["stop"],
                      checkpoint_sha256=sha(directory.path / "checkpoint.pt"))
    except BaseException:
        record["status"], phase["status"] = "interrupted", "interrupted"
        evidence["traceback"] = traceback.format_exc()
        raise
    else:
        complete = all(r["status"] in TERMINAL for r in phase["cases"].values())
        phase["status"] = "learning_closed" if complete else "awaiting_continuation"
        if complete:
            phase["closed_utc"] = datetime.now(timezone.utc).isoformat()
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - start
        evidence["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write(archive / "recovery.json", evidence)
        write(phase_path, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    args = parser.parse_args()
    with (HERE / "runs/recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover(args.decision)


if __name__ == "__main__":
    main()
