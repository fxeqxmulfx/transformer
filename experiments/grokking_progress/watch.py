"""Read current atomic checkpoints; record internals without resuming training.

From python/: uv run --locked python ../experiments/grokking_progress/watch.py
gptmini-seed1. The first run retains observer-v1 source in an isolated
worktree. This observer uses the same saved weights on CPU, explicitly
recorded; it never touches model/optimizer/sampler checkpoints or history.
Only history at or before the saved update enters its phase calculation.
"""

import hashlib
import io
import json
from pathlib import Path
import sys
import time

import torch

from lab.domain.grokking import norm_progress, progress
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.engine.grokking import GrokkingObserver
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code

STUDY = Path(__file__).resolve().parent


def observe(label):
    experiment = dict(load(STUDY).select([label]))[label]
    root = STUDY / "runs" / label
    torch.set_num_threads(1)
    task = ModularTask(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
    model = build_model(experiment.model, task.vocab, experiment.seeds.model)
    observer = GrokkingObserver(experiment.diagnostics, task)
    output = root / "internal_checkpoint_probes.jsonl"
    last = json.loads(output.read_text().splitlines()[-1])["step"] if output.exists() else -1
    identity = code()
    while last < experiment.budget.updates:
        checkpoint = root / "checkpoint.pt"
        if not checkpoint.exists():
            time.sleep(30)
            continue
        data = checkpoint.read_bytes()
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        step = saved["step"]
        if step <= last:
            time.sleep(30)
            continue
        if code() != identity:
            raise RuntimeError("Observer source changed; preserve identity before restarting")
        model.load_state_dict(saved["model"])
        result = observer.observe(model, step)
        history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
        history = [row for row in history if row["step"] <= step]
        history[-1] = {**history[-1], "grokking": result}
        diagnostics = [json.loads(line) for line in (root / "diagnostics.jsonl").read_text().splitlines()]
        norms = norm_progress([row for row in diagnostics if row["step"] <= step])["latest"]
        row = {**result, "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
               "observer_code": identity, "observation_device": "cpu",
               "training_device": experiment.execution.device, "norms": norms,
               "phase_evidence": progress(history)["latest"]}
        with output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(label, step, "S", result["heldout_invariant_energy_fraction"],
              "restricted CE", result["heldout_restricted_loss"],
              "weight norm", norms["groups"]["all"]["parameter_l2"] if norms else None, flush=True)
        last = step


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Supply one declared experiment label")
    observe(sys.argv[1])
