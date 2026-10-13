"""Run the checkpoint repeat and seed controls after the active CUDA run.

From python/: uv run --locked python ../experiments/grokking_progress/queue.py.
The full 150,000-update budgets stay in experiment.py. No optimizer or
experiment setting is supplied on the command line, and the GPU is not
shared with the currently running primary training worker.
"""

import json
from pathlib import Path
import subprocess
import time

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def main():
    result = STUDY / "runs/gptmini-seed1/result.json"
    while not result.exists():
        time.sleep(30)
    completed = json.loads(result.read_text())
    if completed["stop"]["reason"] != "budget" or completed["stop"]["step"] != 150_000:
        raise RuntimeError("Primary CUDA run did not complete its declared budget")
    repeat = STUDY.parent / "grokking_internals"
    status = subprocess.call([str(ROOT / "make.py"), "run", str(repeat)], cwd=ROOT)
    if status:
        raise SystemExit(status)
    raise SystemExit(subprocess.call([str(ROOT / "make.py"), "run", str(STUDY),
                                     "gptmini-seed2", "gptmini-seed3"], cwd=ROOT))


if __name__ == "__main__":
    main()
