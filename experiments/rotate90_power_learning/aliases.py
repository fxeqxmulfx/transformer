"""Check whether failed predictions agree after a further modular power.

This is a descriptive confusion diagnostic, not a causal explanation.
All powers use Python's exact modular arithmetic before tensor lookup.
Run from python/: uv run --locked python ../experiments/rotate90_power_learning/aliases.py
"""

import json
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.rotary_geometry import embedding_geometry
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent
LABELS = ("w16d1-p97-k2", "w64d4-p193-k2", "w16d1-p193-k1", "w16d1-p193-k3")


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    experiments = load(ROOT).experiments
    findings = []
    for label in LABELS:
        experiment = experiments[label]
        run = RunDirectory(ROOT / "runs" / label)
        if run.result() is None:
            continue
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        model = build_model(experiment.model, task.vocab, experiment.seeds.model)
        model.load_state_dict(run.checkpoint("cpu")["model"])
        model.eval()
        result = {"label": label, "prime": task.vocab, "exponent": experiment.benchmark.exponent,
                  "embedding": embedding_geometry(model.E, task.nonzero_order), "heldout": {}}
        rows = task.splits["heldout"]
        predictions = torch.cat([model(chunk[:, :2]).argmax(-1) for chunk in rows.split(1024)])
        for power in (1, 2, 3, 4, 6, 8, 16):
            mapping = torch.tensor([pow(value, power, task.vocab) for value in range(task.vocab)])
            correct = int((mapping[predictions] == mapping[rows[:, 2]]).sum())
            result["heldout"][str(power)] = {"correct": correct, "total": len(rows),
                                             "accuracy": correct / len(rows)}
        findings.append(result)
        print(label, {k: round(v["accuracy"], 5) for k, v in result["heldout"].items()}, flush=True)
    (ROOT / "alias_diagnostics.json").write_text(json.dumps({
        "meaning": "Power j compares prediction^j and target^j modulo P, after training.",
        "scope": "descriptive aliasing; no claim about the training mechanism", "runs": findings}, indent=2) + "\n")


if __name__ == "__main__":
    main()
