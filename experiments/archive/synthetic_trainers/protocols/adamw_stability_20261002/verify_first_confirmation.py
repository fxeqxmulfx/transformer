"""Verify the archived early AdamW prefix without PyTorch or original runs."""

import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.stability_integrity import expected_batches


def verify(directory):
    directory = Path(directory)
    hashes = json.loads((directory / "artifact-hashes.json").read_text())["files"]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    if actual != hashes:
        raise ValueError("Early AdamW prefix artifact hashes differ")
    summary = json.loads((directory / "summary.json").read_text())
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    gradients = [json.loads(line) for line in (directory / "gradients.jsonl").read_text().splitlines()]
    end, config = summary["through_update"], summary["config"]
    if ([p["step"] for p in history] != list(range(0, end + 1, config["eval_every"]))
            or [p["step"] for p in gradients] != list(range(1, end + 1))):
        raise ValueError("Early AdamW prefix is missing or duplicated")
    criterion = PersistenceConfig(**summary["criterion"])
    report = {"plan": {"status": "running", "config": config}, "history": history,
              "completed_steps": end, "final": history[-1]}
    result = assess(report, criterion)
    first = next(p for p in history if p["heldout"]["accuracy"] >= criterion.target)
    if (result != summary["assessment"] or first != summary["first_canonical_heldout_target"]
            or max(gradients, key=lambda p: p["gradient_l2"]) != summary["largest_observed_gradient"]
            or len(history) != summary["history_observations"]
            or len(gradients) != summary["gradient_observations"]):
        raise ValueError("Early AdamW prefix summary differs from measurements")
    for point in gradients:
        batch = expected_batches(config, point["step"])
        if point["batch_size"] != batch["batch_size"] or point["epoch_tail"] != batch["epoch_tail"]:
            raise ValueError("Early gradient trace has wrong batch exposure")
    if result["complete_canonical_history"] or result["stable_grokking"] or result["plateau"] is not None:
        raise ValueError("The early AdamW prefix cannot establish stable grokking")
    return {"verified": True, "through_update": end, "history_observations": len(history),
            "gradient_observations": len(gradients), "phase_eligible": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent / "first-long-confirmation")))
