"""Inspect actual runs and the content-only model's observable input classes.

Source: matchingHeadScores and matchingHeadOutput at Lean commit 83d985f.
In real arithmetic a head at row r depends on its query token and visible
token multiset. Every mixture shares that invariance. Conflicting labels
within such classes lower-bound answer loss independently of head pricing.
The MQAR witness is checked by the lab's original raw-input answer oracle;
its floating-point logits are also compared, without changing any training.

Run from python/: uv run --locked python ../experiments/convex_atomic/analyze.py
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.vocabulary import IGNORE
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

ROOT = Path(__file__).resolve().parent


def signature(tokens, row):
    return tokens[row], tuple(sorted(Counter(tokens[:row + 1]).items()))


def signature_bound(rows):
    """Best possible token accuracy/loss for any function of the physical signature."""
    groups = defaultdict(Counter)
    for tokens, targets, changes in rows.rows.cpu().tolist():
        counts = Counter()
        for query, target in zip(tokens, targets):
            counts[query] += 1
            if target != IGNORE:
                groups[query, tuple(sorted(counts.items()))][target] += 1
    observations = sum(sum(group.values()) for group in groups.values())
    correct = sum(max(group.values()) for group in groups.values())
    loss = sum(-count * math.log(count / sum(group.values()))
               for group in groups.values() for count in group.values()) / observations
    return {"scalar_token_observations": observations, "signature_classes": len(groups),
            "conflicting_classes": sum(len(group) > 1 for group in groups.values()),
            "maximum_token_accuracy_in_real_model": correct / observations,
            "minimum_mean_cross_entropy_in_real_model": loss}


def recall_witness(experiment, task, run):
    """Exchange two table values, preserving visible content and changing an answer."""
    generator = task.generator
    original = generator.sample(random.Random(0), experiment.benchmark.problem.minimum)
    position = next(index for index, target in enumerate(original.targets) if target != IGNORE)
    key = original.tokens[position]
    writes = generator.writes(original.tokens)
    first = next(index for index, (written, value) in enumerate(writes) if written == key)
    other = next(index for index, (written, value) in enumerate(writes) if value != writes[first][1])
    tokens = list(original.tokens)
    a, b = 2 + 2 * first, 2 + 2 * other
    tokens[a], tokens[b] = tokens[b], tokens[a]
    alternative = generator.example(tokens)
    for example in (original, alternative):
        generator.validate(example)
    assert signature(original.tokens, position) == signature(alternative.tokens, position)
    assert original.targets[position] != alternative.targets[position]
    found = {"first_tokens": original.tokens, "second_tokens": alternative.tokens,
             "query_position": position, "first_target": original.targets[position],
             "second_target": alternative.targets[position], "same_physical_signature": True,
             "both_examples_accepted_by_raw_oracle": True}
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    selected = checkpoint.get("best")
    state = checkpoint["model"] if selected is None else selected["model"]
    model = build_model(experiment.model, task.vocab, experiment.seeds.model)
    model.load_state_dict(state)
    # Double precision reduces differences due solely to summation order.
    model.double()
    with torch.no_grad():
        rows = torch.tensor([original.tokens, alternative.tokens])
        positions = torch.full((2, 1), position)
        logits = model(rows, positions)[:, 0]
    return {**found, "selected_checkpoint_step": checkpoint["step"] if selected is None else selected["step"],
            "maximum_logit_difference_float64": float((logits[0] - logits[1]).abs().max()),
            "predictions": logits.argmax(-1).tolist()}


def main():
    torch.set_num_threads(1)
    study, found = load(ROOT), {"runs": {}, "basis_signature_bounds": {}}
    for label, experiment in study.experiments.items():
        run = ROOT / "runs" / label
        result_file = run / "result.json"
        if not result_file.exists():
            found["runs"][label] = {"status": "unfinished"}
            continue
        result = json.loads(result_file.read_text())
        best = result["best"]
        found["runs"][label] = {"status": "finished", "best_step": best["step"],
                                "validation_loss": best["validation"]["loss"],
                                "global_gap_bound_after": result.get("optimizer", {}).get("global_gap_bound_after"),
                                "certificate_step": result["stop"]["step"],
                                "certificate_scope": result.get("optimizer", {}).get("certificate_scope"),
                                "pricing_is_global": result.get("optimizer", {}).get("pricing_is_global")}
        if not label.startswith("basis-"):
            continue
        found["runs"][label].update({"validation_sequence_accuracy": best["validation"]["sequence_accuracy"],
                                    "test_sequence_accuracy": best["test"]["sequence_accuracy"],
                                    "test_loss": best["test"]["loss"], "active_heads": best["atomic"]["active_heads"],
                                    "search_evaluations": result["optimizer"]["search_evaluations"]})
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        found["basis_signature_bounds"][label] = {split: signature_bound(task.splits[split])
                                                   for split in ("validation", "test")}
        if "recall" in label:
            found["recall_binding_counterexample"] = recall_witness(experiment, task, run)
    destination = ROOT / "runs" / "analysis.json"
    destination.write_text(json.dumps(found, indent=2) + "\n")
    print(json.dumps(found, indent=2))


if __name__ == "__main__":
    main()
