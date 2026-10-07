"""Retain immutable copies of atomic checkpoints from both grokking studies.

Run from python/: uv run --locked python ../experiments/grokking_internals/archive.py.
No training file is changed. A whole checkpoint is read once, validated,
and saved with its SHA256. Earlier overwritten checkpoints cannot be
recovered and are never reconstructed from aggregate measurements.
"""

import hashlib
import io
import json
from pathlib import Path
import time

import torch

from lab.infrastructure.loader import load

HERE = Path(__file__).resolve().parent
STUDIES = (HERE, HERE.parent / "grokking_progress")


def retain(root):
    checkpoint = root / "checkpoint.pt"
    if not checkpoint.exists():
        return None
    data = checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    step = saved["step"]
    destination = root / "probe_checkpoints" / f"step-{step:06d}.pt"
    digest = hashlib.sha256(data).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Different checkpoints have the same update: {destination}")
        return step
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    with (destination.parent / "index.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, "checkpoint_sha256": digest,
                                 "bytes": len(data)}, allow_nan=False) + "\n")
    print(root.parent.parent.name, root.name, "retained", step, flush=True)
    return step


def main():
    torch.set_num_threads(1)
    complete = set()
    while True:
        declared = {(study, label): experiment
                    for study in STUDIES for label, experiment in load(study).select([])}
        for (study, label), experiment in declared.items():
            key = (study, label)
            if key in complete:
                continue
            root = study / "runs" / label
            step = retain(root)
            if step == experiment.budget.updates and (root / "result.json").exists():
                complete.add(key)
        if complete == set(declared):
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
