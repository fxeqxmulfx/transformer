"""Exhaustive read-only routing diagnostics and fixed-weight normalizer swaps."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.synthetic_trainers import attention_training, sparsemax_attention
from experiments.synthetic_trainers.attention_layout import current_sources, digest
from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
from experiments.synthetic_trainers.stability_report import write_json


PROTOCOL = Path(__file__).resolve().parent
RAW = Path("experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k")
CERTIFICATE = Path("experiments/synthetic_trainers/baselines/adamw_sparsemax_final_certificate_20261003")


@contextmanager
def observe(normalizer, captured, gradients=False):
    owner = sparsemax_attention if normalizer == "sparsemax" else F
    name = "causal_sparsemax" if normalizer == "sparsemax" else "softmax"
    original = getattr(owner, name)

    def record(scores, *args, **kwargs):
        weights = original(scores, *args, **kwargs)
        if scores.ndim == 4 and scores.shape[-2] == scores.shape[-1]:
            if gradients:
                scores.retain_grad()
            captured.append((scores, weights))
        return weights

    with patch.object(owner, name, record):
        yield


def accumulate(statistics, captured, correct):
    assert len(captured) == 2
    for layer, (scores, weights) in enumerate(captured):
        scores, weights = scores.detach(), weights.detach()
        width = weights.shape[-1]
        future = torch.ones(width, width, dtype=torch.bool).triu(1)
        assert torch.isfinite(scores.masked_select(~future)).all()
        assert not (weights < 0).any() and weights.masked_select(future).eq(0).all()
        assert (weights.sum(-1) - 1).abs().max() < 1e-6
        ordered = scores.masked_fill(future, -torch.inf).topk(2, dim=-1).values
        gaps = ordered[..., 0] - ordered[..., 1]
        support = (weights > 0).sum(-1)
        entropy = -(weights.double() * weights.double().clamp_min(1e-300).log()).sum(-1)
        for head in range(weights.shape[1]):
            for position, label in ((4, "numeric"), (5, "EOS")):
                for flag, group in ((True, "numeric_correct"), (False, "numeric_wrong")):
                    selected = correct == flag
                    count = int(selected.sum())
                    if not count:
                        continue
                    row = statistics[(layer, head, label, group)]
                    row["rows"] += count
                    row["singleton_rows"] += int((support[selected, head, position] == 1).sum())
                    row["strict_unit_gap_rows"] += int((gaps[selected, head, position] > 1).sum())
                    numerator = weights[selected, head, position, 1]
                    denominator = weights[selected, head, position, 3]
                    row["numerator_zero_rows"] += int(numerator.eq(0).sum())
                    row["denominator_zero_rows"] += int(denominator.eq(0).sum())
                    row["both_operand_zero_rows"] += int((numerator.eq(0) & denominator.eq(0)).sum())
                    row["numerator_mass_sum"] += float(numerator.double().sum())
                    row["denominator_mass_sum"] += float(denominator.double().sum())
                    row["entropy_sum"] += float(entropy[selected, head, position].sum())


def inspect_partition(model, normalizer, rows, batch_size):
    statistics = defaultdict(lambda: defaultdict(float))
    numeric_correct = complete_correct = EOS_correct = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = torch.from_numpy(rows[start:start + batch_size].copy())
            captured = []
            with observe(normalizer, captured):
                logits = model(batch[:, :-1])[:, 4:, :]
            predictions = logits.argmax(-1)
            flags = predictions == batch[:, 5:]
            numeric_correct += int(flags[:, 0].sum())
            complete_correct += int(flags.all(1).sum())
            EOS_correct += int(flags[:, 1].sum())
            accumulate(statistics, captured, flags[:, 0])
    result = []
    for (layer, head, position, group), data in sorted(statistics.items()):
        result.append({"layer": layer, "head": head, "position": position, "group": group,
                       **dict(data), "mean_entropy": data["entropy_sum"] / data["rows"]})
    return {"examples": len(rows), "numeric_correct": numeric_correct,
        "complete_RHS_correct": complete_correct, "EOS_correct": EOS_correct,
        "accuracy": complete_correct / len(rows), "routing": result}


def derivative_probe(config, checkpoint, vocabulary, rows):
    batch = torch.from_numpy(rows[:512].copy())
    model = attention_training.make_model(replace(config, device="cpu"), vocabulary)
    model.load_state_dict(checkpoint["model"], strict=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    captured = []
    with observe(config.attention_normalization, captured, gradients=True):
        logits = model(batch[:, :-1])[:, 4:, :]
        target = batch[:, 5:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss.backward()
    observed = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    flags = logits.argmax(-1)[:, 0] == target[:, 0]
    rows_out = []
    for layer, (scores, weights) in enumerate(captured):
        support = (weights.detach() > 0).sum(-1)
        gradient = scores.grad.detach()
        assert torch.isfinite(gradient).all()
        assert gradient.masked_select(weights.detach() == 0).eq(0).all()
        for position, label in ((4, "numeric"), (5, "EOS")):
            for flag, group in ((True, "numeric_correct"), (False, "numeric_wrong")):
                selected = flags == flag
                subset = gradient[selected, :, position, :]
                rows_out.append({"layer": layer, "position": label, "group": group,
                    "rows": subset.shape[0] * subset.shape[1],
                    "singleton_rows": int((support[selected, :, position] == 1).sum()),
                    "entire_zero_score_gradient_rows": int(subset.eq(0).all(-1).sum()),
                    "score_gradient_L2": float(subset.double().norm())})
    model.zero_grad(set_to_none=True)
    ordinary_logits = model(batch[:, :-1])[:, 4:, :]
    ordinary_loss = F.cross_entropy(ordinary_logits.reshape(-1, ordinary_logits.shape[-1]), target.reshape(-1))
    ordinary_loss.backward()
    assert torch.equal(logits.detach(), ordinary_logits.detach()) and torch.equal(loss.detach(), ordinary_loss.detach())
    assert all(torch.equal(observed[name], parameter.grad) and torch.equal(before[name], parameter)
               for name, parameter in model.named_parameters())
    return {"examples": len(batch), "CPU_loss": float(loss.detach()),
        "numeric_correct": int(flags.sum()), "rows": rows_out,
        "all_inactive_score_gradient_entries_exactly_zero": True,
        "instrumented_logits_loss_and_every_parameter_gradient_match": True}


def execute(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Generalization diagnostics require a fresh destination")
    assert not torch.cuda.is_initialized()
    plan = json.loads((PROTOCOL / "attention-pair-plan.json").read_text())
    assert current_sources() == plan["source_hashes"] and attention_training.training_sources() == plan["training_source_hashes"]
    for field in ("Lean_specification_hashes", "papers"):
        assert all(digest(path) == expected for path, expected in plan[field].items())
    torch.set_num_threads(1)
    results, checkpoint_hashes = {}, {}
    for weights_name in ("sparsemax", "softmax"):
        recipe = next(r for r in plan["recipes"] if r["name"] == f"adamw-{weights_name}")
        config = attention_training.AttentionRunConfig(**recipe["config"])
        corpus = make_corpus(config.prime, config.train_fraction, config.data_seed)
        assert corpus.summary() == plan["corpus"]
        case = RAW / f"adamw-{weights_name}"
        checkpoint_path = case / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_hashes[weights_name] = digest(checkpoint_path)
        archive = Path("experiments/synthetic_trainers/baselines") / f"adamw_attention_adamw-{weights_name}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002"
        assert checkpoint["step"] == 300000
        assert checkpoint_hashes[weights_name] == json.loads((archive / "artifact-hashes.json").read_text())["raw_checkpoint_sha256"]
        report = json.loads((case / "measurements.json").read_text())
        for forward in (weights_name, "softmax" if weights_name == "sparsemax" else "sparsemax"):
            model = attention_training.make_model(replace(config, device="cpu", attention_normalization=forward), len(corpus.tokens))
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            assert sum(p.numel() for p in model.parameters()) == 436104 and all(torch.isfinite(p).all() for p in model.parameters())
            key = f"weights-{weights_name}_forward-{forward}"
            result = {partition: inspect_partition(model, forward, rows, config.batch_size)
                      for partition, rows in (("train", corpus.train), ("heldout", corpus.heldout))}
            if forward == weights_name:
                for partition in result:
                    assert result[partition]["accuracy"] == report["final"][partition]["accuracy"]
                result["derivative_probes"] = {partition: derivative_probe(config, checkpoint, len(corpus.tokens), rows)
                    for partition, rows in (("train", corpus.train), ("heldout", corpus.heldout))}
            results[key] = result
            print(json.dumps({"case": key, "counts": {part: {field: result[part][field] for field in
                ("examples", "numeric_correct", "complete_RHS_correct", "EOS_correct", "accuracy")}
                for part in ("train", "heldout")}}), flush=True)
    assert not torch.cuda.is_initialized()
    output.mkdir(parents=True)
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exhaustive_fixed_final_weight_routing_and_forward_swaps; no_training_cause_established",
        "checkpoint_hashes": checkpoint_hashes, "checkpoint_step": 300000,
        "corpus": corpus.summary(), "parameters": 436104, "results": results,
        "derivative_probe_selection": "first_512_rows_in_each_frozen_partition",
        "optimizer_updates_performed": 0, "CUDA_initialized": False,
        "program_sha256": digest(__file__), "all_original_frozen_sources_unchanged": True}
    write_json(output / "observations.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.output)
