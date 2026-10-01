"""Run all requested methods under identical validation stopping settings."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path, DATA_FILE as _data_file
from gpt_mini.infrastructure.benchmark.provenance import git_head as _git_head
from gpt_mini.infrastructure.benchmark.suites import test_suite as _test_suite

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import unittest

import torch

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import log_progress
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results, write_json
from .protocol import check_initial, previous_runs, prior_fingerprints, source_hashes
from .registry import METHODS, PAIRED_NAMES, selected_rates
from .report import write_report
from .stopping import StopConfig
from .storage import write_summary
from .training import train_one
from .validation import validate


def run_tests(directory):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        _test_suite(('optimizer_benchmark',)),
        _test_suite(('magma_benchmark',)),
        _test_suite(('patience_benchmark',)),
        loader.loadTestsFromName("gpt_mini.infrastructure.benchmark.tests.test_parameter_groups")])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"tests_{stamp}.log"
    with path.open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"All CPU/CUDA tests must pass before training: {path}")
    return {"tests": result.testsRun, "failures": 0, "errors": 0, "skipped": 0,
            "log": str(path), "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def launch(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(args.output or (str(_bench_path('experiments/patience_benchmark/results/')) + "/") +
                     ("rtx3050_all" if args.all else "rtx3050_paired"))
    directory.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        torch.set_num_threads(4)
        result = validate(directory, complete=not args.allow_partial)
        write_report(directory)
        print(json.dumps(result, indent=2), flush=True)
        return
    tests = run_tests(directory)
    torch.set_num_threads(4)
    log_progress(directory, f"All {tests['tests']} CPU/CUDA tests passed before training")
    if args.tests_only:
        return
    earlier, old_metadata = previous_runs()
    common = old_metadata["protocol"]
    config = Config(**common["config"])
    data = TextData.load(_data_file, "cuda")
    stop = StopConfig(every=args.every, patience=args.patience, min_delta=args.min_delta,
                      min_steps=args.min_steps, max_steps=args.max_steps,
                      divergence_delta=args.divergence_delta, divergence_patience=args.divergence_patience)
    methods = [m for m in METHODS if args.all or m.name in PAIRED_NAMES]
    rates = selected_rates()
    groups = [(a, m.name) for a in common["attention"] for m in methods]
    random.Random(1729).shuffle(groups)
    jobs = [(a, name, seed) for a, name in groups for seed in common["seeds"]]
    protocol = {"config": asdict(config), "batch": common["batch"], "seeds": common["seeds"],
                "attention": common["attention"], "methods": [asdict(m) for m in methods],
                "selected_rates": {a: {m.name: rates[a][m.name] for m in methods} for a in common["attention"]},
                "stopping": asdict(stop), "data_sha256": data.sha256,
                "data_boundaries": list(data.boundaries), "characters": data.characters,
                "initialization": common["initialization"], "dtype": "float32", "tf32": False,
                "deterministic_algorithms": True, "compile": False, "cuda_graphs": False,
                "selection": "exact best validation checkpoint; test once after stopping",
                "source_hashes": source_hashes(), "prior_artifacts": prior_fingerprints(),
                "job_order": jobs}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    props = torch.cuda.get_device_properties(0)
    metadata = {"protocol": protocol, "protocol_sha256": fingerprint, "tests": tests,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"gpu": props.name, "gpu_total_bytes": props.total_memory,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "git_head": _git_head(),
                    "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                          "--format=csv,noheader"], text=True).strip()}}
    path = directory / "metadata.json"
    if path.exists():
        if json.loads(path.read_text())["protocol_sha256"] != fingerprint:
            raise RuntimeError("Existing output has changed sources/data/stopping; choose a new directory")
    else:
        write_json(path, metadata)
    raw = directory / "runs.jsonl"
    rows = read_results(raw)
    completed = {r["id"] for r in rows}
    artifacts = Path(_bench_path('experiments/runs/patience_benchmark')) / directory.name
    for index, (attention, method, seed) in enumerate(jobs, 1):
        identifier = f"{attention}:{method}:{seed}"
        if identifier in completed:
            continue
        rate = rates[attention][method]
        log_progress(directory, f"run {index}/{len(jobs)} START {identifier} lr={rate:g}")
        row = train_one(config, data, attention, method, rate, seed, common["batch"], stop, artifacts,
                        progress=lambda message: log_progress(directory, message))
        check_initial(row, earlier)
        row["protocol_sha256"] = fingerprint
        with raw.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        rows.append(row)
        completed.add(identifier)
        write_summary(directory, rows, common["seeds"])
        write_report(directory)
        log_progress(directory, f"run {index}/{len(jobs)} END {identifier}: {row['status']} "
                     f"stop={row['stop_reason']} updates={row['actual_steps']} best={row['best_step']} "
                     f"test={row['test_loss']}")
    validate(directory)
    write_report(directory)
    log_progress(directory, "COMPLETE: validation stopping, all-method ranking and all checkpoint audits saved")
