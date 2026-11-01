"""One durable two-hour GPU phase; restarting a process never resets its clock."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "runs/phase.json"


def write(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_budget(reserve=0):
    record = json.loads(LEDGER.read_text())
    left = record["deadline_monotonic"] - time.monotonic() - reserve
    if left <= 0:
        raise TimeoutError("The authorized residual GPU phase is exhausted")
    return left


def start():
    if LEDGER.exists():
        raise FileExistsError("This authorized phase already started")
    LEDGER.parent.mkdir(exist_ok=True, parents=True)
    now = time.monotonic()
    write(LEDGER, {"started_utc": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": now, "deadline_monotonic": now + 7200,
        "wall_budget_seconds": 7200, "status": "profiling", "events": []})


if __name__ == "__main__":
    start()
