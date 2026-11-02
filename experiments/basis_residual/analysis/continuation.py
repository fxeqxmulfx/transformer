"""Explicit validation-only decisions at frozen Basis continuation boundaries.

The original protocol and cohort remain byte-identical. A decision extends
only the update budget, or records an unresolved case as censored with a
reason. This is never a claim of impossibility. The decision, this runner's
hash and every previous endpoint are retained before any continued learning.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import torch

from lab.domain.experiment import require_continuation
from lab.domain.spec import describe, swap
from lab.infrastructure.engine.basis_residual import BasisTraining
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.provenance import provenance
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from protocol import frozen, sha, write

TERMINAL = ("validation99", "numerical_failure", "censored")


def validate_decisions(phase, declaration, decision, protocol_hash):
    if phase["status"] != "awaiting_continuation":
        raise ValueError("Decide only after the complete initial cohort or continuation chunk")
    labels = {case["label"] for case in declaration["cases"]}
    if set(phase["cases"]) != labels or decision["protocol_sha256"] != protocol_hash:
        raise ValueError("Every frozen case and the unchanged protocol must be present")
    unfinished = {label for label, record in phase["cases"].items() if record["status"] not in TERMINAL}
    plans = decision["decisions"]
    if len({plan["label"] for plan in plans}) != len(plans) or {plan["label"] for plan in plans} != unfinished:
        raise ValueError("Decide exactly once for every unfinished case")
    if not plans:
        raise ValueError("There are no unfinished cases")
    for plan in plans:
        record = phase["cases"][plan["label"]]
        if record["status"] != "budget" or record["stop"]["step"] != plan["step"]:
            raise ValueError("A decision must refer to a completed update chunk")
        if record["checkpoint_sha256"] != plan["checkpoint_sha256"]:
            raise ValueError("The decision names a different endpoint")
        if not isinstance(plan.get("reason"), str) or not plan["reason"].strip():
            raise ValueError("Every extension or censoring needs its validation-only reason")
        if plan["action"] == "extend":
            if type(plan.get("updates")) is not int or plan["updates"] <= plan["step"]:
                raise ValueError("A continuation must increase the completed update count")
        elif plan["action"] != "censor" or "updates" in plan:
            raise ValueError("Use an explicit extension or censoring decision")


def continue_cohort(decisions_path):
    declaration = frozen()
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("Frozen runtime changed")
    if (HERE / "results.json").exists():
        raise ValueError("Learning cannot continue after final test has been opened")
    phase_path = HERE / "runs/phase.json"
    phase = json.loads(phase_path.read_text())
    decisions = json.loads(decisions_path.read_text())
    validate_decisions(phase, declaration, decisions, sha(HERE / "protocol.json"))
    study = load(HERE)
    # Validate every referenced state before making the first change.
    for plan in decisions["decisions"]:
        directory = RunDirectory(HERE / "runs" / plan["label"])
        if sha(directory.path / "checkpoint.pt") != plan["checkpoint_sha256"]:
            raise ValueError("A checkpoint changed after the decision")
        state = directory.checkpoint("cpu")
        if state["step"] != plan["step"]:
            raise ValueError("Checkpoint update count differs from the decision")
        spec = swap(study.experiments[plan["label"]], "budget.updates", plan.get("updates", plan["step"]))
        require_continuation(directory.description(), spec)
        history = directory.records("history")
        if history[-1]["step"] != state["step"] or any(spec.benchmark.stable([row]) for row in history):
            raise ValueError("An unfinished run must have an observed endpoint and no earlier crossing")
    evidence = {"decision": decisions, "decision_sha256": sha(decisions_path),
                "runner_source_sha256": sha(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat()}
    record_path = HERE / "runs/continuations" / f"{evidence['decision_sha256']}.json"
    if record_path.exists():
        raise ValueError("This decision was already used; inspect its recorded state before recovery")
    write(record_path, evidence)
    phase.setdefault("continuations", []).append(evidence)
    phase["status"] = "learning"
    write(phase_path, phase)
    for plan in decisions["decisions"]:
        label = plan["label"]
        record = phase["cases"][label]
        directory = RunDirectory(HERE / "runs" / label)
        snapshot = directory.path / "continuation_endpoints" / f"step-{plan['step']:08d}.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(directory.path / "checkpoint.pt", snapshot)
        if sha(snapshot) != plan["checkpoint_sha256"]:
            raise ValueError("The retained continuation endpoint differs from the decision")
        record.setdefault("earlier_endpoints", []).append({
            key: record[key] for key in ("status", "stop", "checkpoint_sha256", "elapsed_seconds")})
        if plan["action"] == "censor":
            record.update(status="censored", censor_reason=plan["reason"])
            write(phase_path, phase)
            continue
        spec = swap(study.experiments[label], "budget.updates", plan["updates"])
        record["status"] = "preparing"
        write(phase_path, phase)
        start = time.monotonic()
        try:
            directory.begin(describe(spec), provenance(spec.execution.device), study.source)
            trainer = BasisTraining(spec, directory, lambda row: print(json.dumps({
                "label": label, "step": row["step"], "validation": row[spec.benchmark.validation_name]["sequence_accuracy"]}),
                flush=True), ResidualCompiledStepper)
            expected = next(case for case in declaration["cases"] if case["label"] == label)
            if trainer.task.fingerprints != expected["data"]["fingerprints"]:
                raise ValueError("Frozen Basis data changed")
            record.update(status="learning", preparation_seconds=time.monotonic() - start)
            write(phase_path, phase)
            try:
                result = trainer()
                record.update(status=result["stop"]["reason"], stop=result["stop"])
            except FloatingPointError:
                record.update(status="numerical_failure", traceback=traceback.format_exc())
            record.update(elapsed_seconds=time.monotonic() - start,
                          checkpoint_sha256=sha(directory.path / "checkpoint.pt"))
            del trainer
            torch.cuda.empty_cache()
        except Exception:
            record.update(status="continuation_error", traceback=traceback.format_exc())
            phase["status"] = "continuation_error"
            write(phase_path, phase)
            raise
        write(phase_path, phase)
    complete = all(record["status"] in TERMINAL for record in phase["cases"].values())
    phase["status"] = "learning_closed" if complete else "awaiting_continuation"
    if complete:
        phase["closed_utc"] = datetime.now(timezone.utc).isoformat()
    write(phase_path, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path)
    continue_cohort(parser.parse_args().decisions)
