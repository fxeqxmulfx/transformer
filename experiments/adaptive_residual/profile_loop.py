"""Profile each warmed training case with Python3.15's parent-owned sampler.

The sampler reads the child process for ten seconds after its first 1,000-step
observation. It changes no model, optimizer, sampler or stopping state. This
measures Python wall-stack residency, not individual CUDA kernel duration.
Every case receives the same sampling window; retain failures and boundaries.
"""

from contextlib import redirect_stdout
import json
from pathlib import Path
import platform
from profiling.sampling import PstatsCollector
from profiling.sampling.constants import PROFILING_MODE_WALL
from profiling.sampling.sample import sample
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def current():
    phase = read(HERE / "runs/phase.json")
    return next((label for label, record in phase.get("cases", {}).items()
                 if record["status"] == "learning"), None)


def latest(label):
    try:
        return json.loads((HERE / "runs" / label / "history.jsonl").read_text().splitlines()[-1])["step"]
    except (FileNotFoundError, IndexError, json.JSONDecodeError):
        return 0


def profile(pid, label, folder):
    record = {"label": label, "python": platform.python_version(), "step_before": latest(label),
              "requested_seconds": 10, "interval_microseconds": 1000,
              "mode": "nonblocking main-thread wall sampling with native call markers"}
    collector = PstatsCollector(1000)
    started = time.monotonic()
    try:
        sample(pid, collector, duration_sec=10, mode=PROFILING_MODE_WALL, native=True)
        collector.export(str(folder / f"{label}.pstats"))
        with (folder / f"{label}.txt").open("w") as output, redirect_stdout(output):
            collector.print_stats(sort=0, limit=25, mode=PROFILING_MODE_WALL)
        rows = [{"file": key[0], "line": key[1], "function": key[2], **value}
                for key, value in collector.result.items()]
        record.update(status="complete", top_direct=sorted(rows, key=lambda row: row["direct_calls"], reverse=True)[:25],
                      top_cumulative=sorted(rows, key=lambda row: row["cumulative_calls"], reverse=True)[:25],
                      direct_samples=sum(row["direct_calls"] for row in rows))
    except Exception:
        # A sampling failure must not terminate the user's promising training run.
        record.update(status="failed", traceback=traceback.format_exc())
    record.update(elapsed_seconds=time.monotonic() - started, step_after=latest(label),
                  label_after=current())
    record["single_case_window"] = record["label_after"] == label
    (folder / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("label", "status", "step_before", "step_after")}), flush=True)


def run():
    folder = HERE / "runs/profiles"
    folder.mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/training.log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "cohort.py"), "run"], stdout=log, stderr=subprocess.STDOUT)
        print(f"Training child PID {process.pid}; log {HERE / 'runs/training.log'}", flush=True)
        while process.poll() is None:
            label = current()
            if label is not None and latest(label) >= 1000 and not (folder / f"{label}.json").exists():
                profile(process.pid, label, folder)
            time.sleep(1)
        status = process.wait()
    raise SystemExit(status)


if __name__ == "__main__":
    run()
