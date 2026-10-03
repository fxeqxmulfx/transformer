"""Frozen native checkpoint CPU routing/gradient probe, with no optimizer update.

The active scientific source/model/criterion stay unchanged. This diagnostic
describes an actual floating-point sparsemax execution; it does not prove why
generalization fails or substitute for the fresh trained softmax control.
"""

import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.synthetic_trainers import attention_training, sparsemax_attention
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.paper_reproduction.batches import next_batch
from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
from experiments.synthetic_trainers.runtime import write_json


PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
RAW = Path("experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k")


def support_histogram(counts, width):
    return {str(k): int(n) for k, n in enumerate(torch.bincount(counts.flatten(), minlength=width + 1)) if n}


def probe(config, checkpoint, vocabulary, batch, *, instrumented):
    model = attention_training.make_model(replace(config, device="cpu"), vocabulary)
    model.load_state_dict(checkpoint["model"])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    captured = []
    original = sparsemax_attention.causal_sparsemax

    def record(scores):
        weights = original(scores)
        scores.retain_grad()
        captured.append((scores, weights))
        return weights

    model.train(); model.zero_grad(set_to_none=True)
    with patch.object(sparsemax_attention, "causal_sparsemax", record if instrumented else original):
        logits = model(batch[:, :-1])[:, 4:, :]
        target = batch[:, 5:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    if any(not torch.equal(before[name], p) for name, p in model.named_parameters()):
        raise ValueError("The diagnostic changed checkpoint parameters")
    if any(not torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("The frozen-checkpoint gradient is nonfinite")
    layers = []
    for index, (scores, weights) in enumerate(captured):
        weights = weights.detach()
        width = weights.shape[-1]
        future = torch.ones(width, width, dtype=torch.bool).triu(1)
        support = (weights > 0).sum(-1)
        row_error = float((weights.sum(-1) - 1).abs().max())
        future_mass = float(weights.masked_select(future).abs().max())
        if (not torch.isfinite(scores).all() or not torch.isfinite(scores.grad).all()
                or (weights < 0).any() or row_error > 1e-6 or future_mass != 0):
            raise ValueError("Actual routing violates finite causal simplex checks")
        singleton = support == 1
        singleton_gradient = scores.grad[singleton]
        if not torch.equal(singleton_gradient, torch.zeros_like(singleton_gradient)):
            raise ValueError("The implemented singleton-support score derivative is not zero")
        q, k, v = gradients[f"blocks.{index}.attn.qkv.weight"].chunk(3, dim=0)
        supervised = support[:, :, 4:]
        layers.append({"layer": index, "all_prefix_rows": support.numel(),
            "all_prefix_support_histogram": support_histogram(support, width),
            "directly_supervised_rows": supervised.numel(),
            "directly_supervised_support_histogram": support_histogram(supervised, width),
            "directly_supervised_singleton_fraction": float((supervised == 1).float().mean()),
            "causal_future_mass_max": future_mass, "simplex_row_sum_error_max": row_error,
            "all_singleton_score_gradient_entries_exactly_zero": True,
            "all_score_gradient_L2": float(scores.grad.double().norm()),
            "Q_weight_gradient_L2": float(q.double().norm()),
            "K_weight_gradient_L2": float(k.double().norm()),
            "V_weight_gradient_L2": float(v.double().norm()),
            "log_alpha_gradient": gradients[f"blocks.{index}.attn.log_alpha"].tolist(),
            "inverse_temperatures": model.blocks[index].attn.log_alpha.detach().exp().tolist()})
    return {"CPU_RHS_loss": float(loss.detach()), "CPU_RHS_accuracy": float((logits.argmax(-1) == target).all(1).float().mean()),
            "batch_size": len(batch), "layers": layers}, gradients, logits.detach().clone()


def execute(output, snapshot, checkpoint_source=None):
    output, snapshot = Path(output), Path(snapshot)
    if output.exists() or snapshot.exists():
        raise FileExistsError("Routing diagnostic needs fresh archive and local checkpoint snapshot")
    torch.set_num_threads(1)
    manifest = json.loads((RAW / "plan.json").read_text())
    case = RAW / "adamw-sparsemax"
    config = attention_training.AttentionRunConfig(**next(r for r in manifest["recipes"] if r["name"] == "adamw-sparsemax")["config"])
    for filename, expected in manifest["source_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("The frozen active scientific source changed")
    for filename, expected in manifest["Lean_specification_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("The frozen Lean specification changed")
    if digest(PROTOCOL / "sparsemax-preparation/validation.json") != manifest["Lean_audit_validation_sha256"]:
        raise ValueError("The preserved Lean audit changed")
    source = Path(checkpoint_source) if checkpoint_source is not None else case / "checkpoint.pt"
    blob = source.read_bytes()
    snapshot.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(blob)
    if digest(snapshot) != digest(source):
        snapshot.unlink()
        raise RuntimeError("Checkpoint changed during capture; retry the same read-only snapshot")
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    states = list(checkpoint["optimizer"]["state"].values())
    if any(int(state["step"]) != checkpoint["step"] or set(state) != {"step", "exp_avg", "exp_avg_sq"}
           for state in states):
        raise ValueError("The captured native checkpoint state differs from the completed updates")
    corpus = make_corpus(config.prime, config.train_fraction, config.data_seed)
    if corpus.summary() != manifest["corpus"]:
        raise ValueError("The frozen diagnostic corpus changed")
    count_model = attention_training.make_model(replace(config, device="cpu"), len(corpus.tokens))
    parameters = sum(p.numel() for p in count_model.parameters())
    if parameters != 436104:
        raise ValueError("The diagnostic scientific model parameter count changed")
    del count_model
    generator = torch.Generator(); generator.set_state(checkpoint["batch_generator_state"])
    permutation, cursor = checkpoint["permutation"], checkpoint["cursor"]
    batches = []
    for offset in range(1, 30):
        selected, permutation, cursor, metadata = next_batch(permutation, cursor, generator, config.batch_size, config.batch_policy)
        if offset == 1 or metadata["epoch_tail"]:
            batches.append((offset, selected, metadata))
        if metadata["epoch_tail"]:
            break
    results = []
    for offset, selected, metadata in batches:
        batch = torch.from_numpy(corpus.train[selected.numpy()].copy())
        result, gradients, logits = probe(config, checkpoint, len(corpus.tokens), batch, instrumented=True)
        original, expected, expected_logits = probe(config, checkpoint, len(corpus.tokens), batch, instrumented=False)
        if (result["CPU_RHS_loss"] != original["CPU_RHS_loss"] or result["CPU_RHS_accuracy"] != original["CPU_RHS_accuracy"]
                or not torch.equal(logits, expected_logits)
                or any(not torch.equal(value, expected[name]) for name, value in gradients.items())):
            raise ValueError("Routing observation altered an actual model derivative")
        results.append({"sampling_stream_step": checkpoint["step"] + offset, "fixed_weight_step": checkpoint["step"],
                        "selected_training_row_indices": selected.tolist(), "batch_metadata": metadata,
                        "uninstrumented_logits_loss_and_all_parameter_gradients_exactly_match": True, **result})
    if torch.cuda.is_initialized():
        raise ValueError("The CPU diagnostic unexpectedly initialized CUDA")
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "probe-snapshot.py")
    write_json(output / "frozen-root-plan.json", manifest)
    write_json(output / "routing-gradient-probes.json", {"scope": "fixed_checkpoint_CPU_derivatives; no_causal_learning_claim",
        "checkpoint_step": checkpoint["step"], "checkpoint_sha256": digest(snapshot), "local_checkpoint_path": str(snapshot),
        "parameters": parameters, "CPU_only": True,
        "CUDA_initialized": False, "optimizer_updates_performed": 0,
        "later_tail_batch_weights_held_fixed": True, "native_optimizer_state_step": checkpoint["step"], "probes": results})
    write_json(output / "artifact-hashes.json", {"files": {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact-hashes.json"}})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path, help="Replay a saved checkpoint without reading the live trainer")
    args = parser.parse_args()
    rows = execute(args.output, args.snapshot, args.checkpoint_source)
    print(json.dumps({"CPU_probes": len(rows), "checkpoint_step": rows[0]["fixed_weight_step"],
                      "optimizer_updates": 0, "routing_and_uninstrumented_gradients_match": True}))
