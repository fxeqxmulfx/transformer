"""Compare binding runs and construct a compact physical recall-capacity witness.

Source: pairedHeadOutput and pairedBinding_fit. Signed permutations of
(1,2,3,4) provide 384 different integer vectors in [-4,4]^4 with squared
norm 30. Distinct vectors have inner product at most 29, hence actual
sparsemax routes exactly to the value following a matching key. The probe
uses total width eight and cap four, just like both trained Basis arms.
Its Q/K and identity values are constructed, not learned. No probe state
is passed to the actual training runs or their numerical head search.

Run from python/: uv run --locked python ../experiments/convex_binding/analyze.py
"""

from itertools import permutations, product
import json
import math
from pathlib import Path
import random

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.vocabulary import IDENTITY_BASE, IGNORE
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.atomic import MatchingHead, paired_head_forward

ROOT = Path(__file__).resolve().parent


def recall_capacity(experiment, task):
    """Test a constructed feasible head against held-out raw-oracle answers."""
    symbols, cap, width = experiment.benchmark.task.symbols, experiment.model.cap, experiment.model.width
    assert width == 8 and cap == 4 and experiment.benchmark.task.overwrites == 0
    codes = torch.tensor([[sign * number for sign, number in zip(signs, order)]
                          for order in permutations((1, 2, 3, 4))
                          for signs in product((-1, 1), repeat=4)], dtype=torch.float64)
    assert len(codes.unique(dim=0)) == 384 and symbols <= len(codes)
    assert (codes.square().sum(1) == 30).all()
    differences = 30 - codes @ codes.T
    assert differences[~torch.eye(384, dtype=torch.bool)].min() == 1
    # Match pairedRecallCode's exact permutation/sign decoder in Lean.
    selected = []
    for index in range(256):
        permutation, row = index // 16, []
        for coordinate in range(4):
            slot = ((coordinate + permutation // 2 % 2) % 2 if coordinate < 2
                    else 2 + (coordinate - 2 + permutation % 2) % 2)
            magnitude = 1 + (slot + permutation // 4) % 4
            row.append(-magnitude if index % 16 >> coordinate & 1 else magnitude)
        selected.append(row)
    selected = torch.tensor(selected, dtype=torch.float64)
    assert len(selected.unique(dim=0)) == 256 and (selected.square().sum(1) == 30).all()
    assert (30 - selected @ selected.T)[~torch.eye(256, dtype=torch.bool)].min() == 1
    q, k = (torch.zeros(task.vocab, width, dtype=torch.float64) for _ in range(2))
    keys = torch.arange(IDENTITY_BASE, IDENTITY_BASE + symbols)
    q[keys, 4:] = k[keys, 4:] = selected[:symbols]
    values = torch.full((task.vocab, task.vocab), -cap, dtype=torch.float64)
    labels = torch.arange(IDENTITY_BASE + symbols, task.vocab)
    values[labels, labels] = cap
    head = MatchingHead(q, k, values, forward_map=paired_head_forward)
    assert max(float(t.detach().abs().max()) for t in head.parameters()) <= cap
    found = {"constructed_not_trained": True, "same_width_and_cap_as_training": True,
             "width": width, "cap": cap, "active_heads": 1, "available_key_codes": 384,
             "selected_key_codes": symbols, "codes_match_lean_decoder": True,
             "minimum_distinct_key_score_gap": 1,
             "universal_bounded_logit_cross_entropy_floor": math.log1p((task.vocab - 1) * math.exp(-2 * cap)),
             "splits": {}}
    with torch.no_grad():
        for split in ("validation", "test"):
            rows = task.splits[split].rows
            correct, exact, count, loss = 0, 0, 0, 0.0
            for chunk in rows.split(32):
                tokens, targets = chunk[:, 0], chunk[:, 1]
                valid = targets != IGNORE
                positions = valid.int().argsort(dim=1, descending=True, stable=True)[:, :8]
                logits = head(tokens, positions)
                answers = targets.gather(1, positions)
                hit = logits.argmax(-1) == answers
                correct += int(hit.sum())
                exact += int(hit.all(1).sum())
                count += answers.numel()
                loss += float(task.loss(logits, answers)) * answers.numel()
            found["splits"][split] = {"tokens": count, "examples": len(rows),
                                       "token_accuracy": correct / count,
                                       "sequence_accuracy": exact / len(rows), "loss": loss / count}
    found["binding_swap"] = swap_witness(experiment, task, head)
    return found


def swap_witness(experiment, task, model):
    """Change an oracle answer by exchanging values, then measure actual logits."""
    generator = task.generator
    original = generator.sample(random.Random(0), experiment.benchmark.problem.minimum)
    position = next(i for i, target in enumerate(original.targets) if target != IGNORE)
    writes = generator.writes(original.tokens)
    first = next(i for i, (key, value) in enumerate(writes) if key == original.tokens[position])
    second = next(i for i, (key, value) in enumerate(writes) if value != writes[first][1])
    tokens = list(original.tokens)
    a, b = 2 + 2 * first, 2 + 2 * second
    tokens[a], tokens[b] = tokens[b], tokens[a]
    alternative = generator.example(tokens)
    for example in (original, alternative):
        generator.validate(example)
    with torch.no_grad():
        logits = model.double()(torch.tensor([original.tokens, alternative.tokens]),
                                torch.full((2, 1), position))[:, 0]
    return {"raw_oracle_accepted": True, "query_position": position,
            "targets": [original.targets[position], alternative.targets[position]],
            "maximum_logit_difference_float64": float((logits[0] - logits[1]).abs().max()),
            "predictions": logits.argmax(-1).tolist()}


def softmax_reference(split_fingerprints):
    """Compare archived GPTMini runs on identical data without equating FLOPs."""
    baseline = ROOT.parent / "basis" / "runs"
    records = {}
    for seed in range(3):
        label = f"easy-small-recall-seed{seed}"
        directory = baseline / label
        result = json.loads((directory / "result.json").read_text())
        assert result["split_fingerprints"] == split_fingerprints
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        at500 = next(row for row in history if row["step"] == 500)
        records[label] = {
            "same_split_fingerprints": True,
            "solved_step": result["stop"]["step"],
            "wall_seconds": result["wall_seconds"],
            "validation_sequence_accuracy": result["best"]["validation"]["sequence_accuracy"],
            "test_sequence_accuracy": result["best"]["test"]["sequence_accuracy"],
            "test_token_accuracy": result["best"]["test"]["token_accuracy"],
            "step500_validation_sequence_accuracy": at500["validation"]["sequence_accuracy"],
            "step500_validation_token_accuracy": at500["validation"]["token_accuracy"],
        }
    return {"architecture": "GPTMini: width 64, depth 2, four softmax heads, QKNorm, XSA, FFN, RoPE",
            "execution": "Compiled CPU, one thread; archived October 4 runs",
            "equal_flops_measured": False, "runs": records}


def main():
    torch.set_num_threads(4)
    study = load(ROOT)
    found = {"runs": {}}
    for label, experiment in study.experiments.items():
        run = ROOT / "runs" / label
        result_file = run / "result.json"
        if not result_file.exists():
            found["runs"][label] = {"status": "unfinished"}
            continue
        result = json.loads(result_file.read_text())
        best = result["best"]
        optimizer = result.get("optimizer", {})
        record = {"status": "finished", "best_step": best["step"],
                  "validation_loss": best["validation"]["loss"],
                  "outer_updates": result["stop"]["step"],
                  "wall_seconds": result["wall_seconds"],
                  "training_seconds": result["training_seconds"],
                  "search_evaluations": optimizer.get("search_evaluations"),
                  "global_gap_bound_after": optimizer.get("global_gap_bound_after"),
                  "certificate_scope": optimizer.get("certificate_scope"),
                  "active_heads": best["atomic"]["active_heads"]}
        found["runs"][label] = record
        if label.startswith("basis-"):
            record.update({"test_sequence_accuracy": best["test"]["sequence_accuracy"],
                           "test_token_accuracy": best["test"]["token_accuracy"],
                           "test_loss": best["test"]["loss"]})
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
            task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
            model = build_model(experiment.model, task.vocab, experiment.seeds.model)
            model.load_state_dict(checkpoint["best"]["model"])
            record["binding_swap"] = swap_witness(experiment, task, model)
    experiment = study.experiments["basis-paired-recall-seed0"]
    task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
    found["compact_recall_capacity"] = recall_capacity(experiment, task)
    result = json.loads((ROOT / "runs" / "basis-paired-recall-seed0" / "result.json").read_text())
    found["softmax_reference"] = softmax_reference(result["split_fingerprints"])
    (ROOT / "runs" / "analysis.json").write_text(json.dumps(found, indent=2) + "\n")
    print(json.dumps(found, indent=2))


if __name__ == "__main__":
    main()
