"""Use only the reserved end of task145's existing GPU phase for capture checks."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1] / "experiments/residual_generalization"
LEDGER = HERE / "runs/phase.json"


def main():
    while True:
        phase = json.loads(LEDGER.read_text())
        if phase["status"] in ("learning_closed", "closed"):
            break
        if time.monotonic() >= phase["deadline_monotonic"]:
            raise TimeoutError("No diagnostic may extend the existing GPU budget")
        time.sleep(5)
    deadline = phase["deadline_monotonic"] - 10
    records = []
    for mode in ("original", "fixed_batch"):
        remaining = deadline - time.monotonic()
        allowance = min(45, remaining - 120) if mode == "original" else min(120, remaining)
        if allowance < 15:
            records.append({"mode": mode, "status": "not_started", "remaining_seconds": remaining})
            continue
        record = {"mode": mode, "started_monotonic": time.monotonic(), "allowance_seconds": allowance}
        path = HERE / "runs" / f"capture-{mode}.log"
        with path.open("w") as output:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("residual_capture_probe.py")), mode],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = child.wait(timeout=allowance)
                record.update(status="passed" if status == 0 else "failed", returncode=status)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=2)
                record.update(status="timed_out", returncode=child.returncode)
        record["ended_monotonic"] = time.monotonic()
        records.append(record)
        print(json.dumps(record), flush=True)
    (HERE / "runs/capture-checks.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
