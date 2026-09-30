"""Matched MQAR training: ordinary RoPE Transformer versus sparsemax.

The ordinary completed results are reused after checking their configuration
and split identities. Only the attention normalization changes in the new
model. Sparsemax trains first; missing ordinary comparisons are resumed from
their saved checkpoints after the new sweep, without overlapping GPU jobs.
"""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import time

import torch

from . import benchmark
from .config import profile
from .data import load_split, validate_batch
from .engine import evaluate, train_run, write_json
from .runtime import record_execution
from .sparse_attention import SparsemaxTransformer


def run(config, output, data_root, baseline, compiled=False):
    output, baseline = Path(output), Path(baseline)
    if output.resolve() == baseline.resolve():
        raise ValueError("Sparsemax and baseline need separate output directories")
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.loads(json.dumps(config.to_dict()))
    metadata = {"config": serialized, "attention": "causal_sparsemax",
                "baseline": str(baseline.resolve())}
    with (output / ".lock").open("a") as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already using this run directory") from None
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != metadata:
            raise RuntimeError("The output directory belongs to a different configuration")
        write_json(config_path, metadata)
        baseline.mkdir(parents=True, exist_ok=True)
        with (baseline / ".lock").open("a") as baseline_lock:
            try:
                fcntl.flock(baseline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The baseline is active; stop it before sharing its GPU") from None
            comparison = baseline / "comparison.json"
            reference = json.loads(comparison.read_text()) if comparison.exists() else None
            if reference is not None and reference["config"] != serialized:
                raise RuntimeError("The baseline has a different training configuration")
            if reference is None:
                reference = {"config": serialized, "results": []}
            # Freeze the baseline report used by this comparison, including its
            # history of eager/compiled execution and its original source hashes.
            write_json(output / "baseline.json", reference)
        return run_sparsemax(config, output, data_root, reference, compiled, baseline)


def run_sparsemax(config, output, data_root, reference, compiled, baseline):
    torch.set_num_threads(6)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this experiment does not fall back to CPU")
    environment, history = record_execution(
        output, config, compiled, extra_files=("sparse_attention.py", "attention_ablation.py"))
    print(json.dumps({"event": "start", "attention": "causal_sparsemax",
                      "environment": environment}), flush=True)
    comparison_path = output / "comparison.json"
    report = json.loads(comparison_path.read_text()) if comparison_path.exists() else {
        "config": reference["config"], "results": [],
        "scope": "Same learned Transformer; replace only causal softmax with sparsemax",
        "selection": "Validation accuracy then loss, separately for each model; no test selection",
        "convexity": "Weight-row inference at fixed scores; joint model training is nonconvex",
        "numerics": "Softmax uses fused SDPA; sparsemax uses explicit FP32 scores and projection"}
    report.update(environment=environment, execution_history=history,
                  baseline_sha256=hashlib.sha256((output / "baseline.json").read_bytes()).hexdigest())
    write_json(comparison_path, report)
    completed = {(row["length"], row["width"]) for row in report["results"]}
    ordinary = {(row["length"], row["width"]): row for row in reference["results"]}
    for length in config.lengths:
        if all((length, width) in completed for width in config.widths):
            continue
        data, identities = {}, {}
        for split in ("train", "validation", "test"):
            tensors, identities[split] = load_split(data_root, config, length, split)
            for start in range(0, len(tensors[0]), 1024):
                validate_batch(*(t[start:start + 1024] for t in tensors), config.vocab)
            data[split] = tuple(t.to(config.device) for t in tensors)
        if len(set(identities.values())) != 3:
            raise AssertionError("Data splits are not distinct")
        for width in config.widths:
            if (length, width) in completed:
                continue
            baseline_row = ordinary.get((length, width))
            if baseline_row is not None and baseline_row["splits"] != identities:
                raise RuntimeError("The baseline used different data splits")
            candidates = []
            for rate in config.learning_rates:
                directory = output / f"n{length}-d{width}-lr{rate:.8g}"
                result, checkpoint = train_run(
                    config, length, width, rate, data["train"], data["validation"],
                    directory, output / "status.json", compiled=compiled,
                    model_factory=SparsemaxTransformer)
                candidates.append((result, checkpoint))
            selected, checkpoint = max(candidates, key=lambda item: (
                item[0]["validation"]["accuracy"], -item[0]["validation"]["loss"]))
            model = SparsemaxTransformer(config.vocab, width, config.layers, config.heads,
                                        config.mlp_ratio, config.rope_base).to(config.device)
            model.load_state_dict(torch.load(checkpoint, map_location=config.device,
                                            weights_only=True))
            test = evaluate(model, data["test"], config.batch_size(length, width), config)
            if baseline_row is not None and selected["parameters"] != baseline_row["selected_run"]["parameters"]:
                raise AssertionError("The models have different parameter counts")
            row = {"length": length, "width": width, "pairs": length // 4,
                   "splits": identities, "softmax": None if baseline_row is None else baseline_row["rope"],
                   "sparsemax": test,
                   "softmax_selected_run": None if baseline_row is None else baseline_row["selected_run"],
                   "sparsemax_selected_run": selected,
                   "accuracy_difference": None if baseline_row is None else
                       test["accuracy"] - baseline_row["rope"]["accuracy"]}
            report["results"].append(row)
            write_json(comparison_path, report)
            print(json.dumps({"event": "comparison", **row}), flush=True)
            del model
        del data
        if config.device == "cuda":
            torch.cuda.empty_cache()
    if any(row["softmax"] is None for row in report["results"]):
        write_json(output / "status.json", {"event": "resuming_missing_baseline",
                                           "time_unix": time.time()})
        reference = benchmark.run(config, baseline, Path(data_root), compiled)
        ordinary = {(row["length"], row["width"]): row for row in reference["results"]}
        for row in report["results"]:
            if row["softmax"] is not None:
                continue
            original = ordinary[row["length"], row["width"]]
            if original["splits"] != row["splits"] or (
                    original["selected_run"]["parameters"] != row["sparsemax_selected_run"]["parameters"]):
                raise RuntimeError("The resumed baseline does not match the sparsemax experiment")
            row.update(softmax=original["rope"], softmax_selected_run=original["selected_run"],
                       accuracy_difference=row["sparsemax"]["accuracy"] - original["rope"]["accuracy"])
        write_json(output / "baseline.json", reference)
        report["baseline_sha256"] = hashlib.sha256((output / "baseline.json").read_bytes()).hexdigest()
        write_json(comparison_path, report)
    write_json(output / "status.json", {"event": "complete", "time_unix": time.time()})
    return report


def main():
    parser = argparse.ArgumentParser(description="Replace only RoPE attention normalization")
    parser.add_argument("--profile", choices=("full", "capacity", "smoke", "sanity"), default="full")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/sparsemax"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/full"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = replace(profile(args.profile), device=args.device,
                     precision="bf16" if args.device == "cuda" else "fp32")
    run(config, args.output, args.data_root, args.baseline, args.compile)


if __name__ == "__main__":
    main()
