"""One frozen first pass and a shared-time continuation round under task145's cap.

No outcome changes a recipe. Stable runs and runs at their update cap finish;
only complete, time-censored states enter the predeclared continuation round.
The remaining time is shared equally among its remaining cases in fixed order.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lab.infrastructure.provenance import code
from phase import LEDGER, require_budget, write

HERE = Path(__file__).resolve().parent


def run_case(declaration, record, allowance=None):
    label = declaration["label"]
    suffix = "first" if allowance is None else "continuation"
    case = {"label": label, "round": suffix, "status": "not_started", "started_monotonic": time.monotonic()}
    record["cases"].append(case)
    try:
        timeout = require_budget(180)
    except TimeoutError:
        write(LEDGER, record)
        return case
    command = [sys.executable, str(HERE / "run_case.py"), label]
    if allowance is not None:
        case["allowance_seconds"] = allowance
        command.append(str(allowance))
    case["status"] = "running"
    write(LEDGER, record)
    print(json.dumps(case), flush=True)
    with (HERE / "runs" / f"{label}-{suffix}.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = child.wait(timeout=max(0, timeout - 5))
            case.update(returncode=result, status="completed" if result == 0 else "failed")
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            case.update(returncode=child.returncode, status="global_budget_censored")
    case["ended_monotonic"] = time.monotonic()
    result_path = HERE / "runs" / label / "result.json"
    if result_path.exists():
        case["stop"] = json.loads(result_path.read_text())["stop"]
        if case["status"] == "completed":
            case["status"] = case["stop"]["reason"]
    write(LEDGER, record)
    print(json.dumps(case), flush=True)
    return case


def main():
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["engine_sources"] != code():
        raise ValueError("Frozen engine source hashes changed")
    for name, digest in protocol["cohort_sources"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen cohort source changed: {name}")
    record = json.loads(LEDGER.read_text())
    if record["status"] != "profiling":
        raise ValueError("The learning phase has already been launched")
    record.update(status="learning", protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), cases=[])
    write(LEDGER, record)
    unfinished = []
    for declaration in protocol["cases"]:
        case = run_case(declaration, record)
        if case["status"] == "wall_budget_censored" and case["stop"]["step"] < declaration["experiment"]["budget"]["updates"]:
            unfinished.append(declaration)
    for index, declaration in enumerate(unfinished):
        try:
            # Leave compilation overhead within each equal share, and 180s globally.
            allowance = max(0., require_budget(180) / (len(unfinished) - index) - 30.)
        except TimeoutError:
            break
        if allowance < 30:
            break
        run_case(declaration, record, allowance)
    record.update(status="learning_closed", learning_closed_monotonic=time.monotonic())
    write(LEDGER, record)


if __name__ == "__main__":
    main()
