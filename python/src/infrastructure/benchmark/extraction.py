#!/usr/bin/env python3
"""Encode the completed optimizer benchmark's logged binary64 values in Lean.

The Lean kernel checks arithmetic on this table. It does not verify the GPU
execution that produced the input. --check validates the extraction boundary
without rewriting the generated file. Run from any working directory.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from .paths import ARCHIVE_ROOT, archive_path


ROOT = ARCHIVE_ROOT
LOG = ROOT / "experiments/optimizer_benchmark/results/rtx3050/runs.jsonl"
OUT = ROOT / "src/Transformer/OptimizerBenchmark/Basic.lean"
PROTOCOL = "566657d26031eb933ac6f3ea1897691c4077ce77e03f37956f39760d72b8e801"
ATTENTION = ("softmax", "sparsemax")
METHODS = (
    "sgd", "adagrad", "adam", "adamw", "amsgrad", "amsgrad_inverse",
    "amsgrad_geometric", "adamx", "adamnc", "muon", "muon_guarded",
    "dash_evd", "dash_ndb", "dash_cn", "dash_chebyshev", "dash_ndb_guarded",
    "adafisher", "adafisherw",
)


def extract():
    raw = LOG.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    final = [r for r in records if r["phase"] == "final"]
    expected = {(a, m, s) for a in ATTENTION for m in METHODS for s in range(3)}
    actual = [(r["attention"], r["method"], r["seed"]) for r in final]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected exactly one final run per attention/method/seed")
    table = {}
    for r, key in zip(final, actual):
        if (r["status"] != "ok" or r["steps"] != 1000 or
                r["completed_steps"] != 1000 or r["protocol_sha256"] != PROTOCOL):
            raise ValueError(f"Incomplete or incompatible record: {key}")
        if not isinstance(r["test_loss"], float) or not math.isfinite(r["test_loss"]):
            raise ValueError(f"Invalid logged loss: {key}")
        table[key] = Fraction.from_float(r["test_loss"])
    # Check the shared experiment setup at the extraction boundary.
    for s in range(3):
        group = [r for r in final if r["seed"] == s]
        for field in ("initial_sha256", "batch_sha256"):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f"Unpaired seed {s}: {field}")
    for a in ATTENTION:
        for m in METHODS:
            if len({r["lr"] for r in final if r["attention"] == a and r["method"] == m}) != 1:
                raise ValueError(f"Inconsistent selected rate: {a}/{m}")
    summary = json.loads((LOG.parent / "summary.json").read_text())
    rows = summary["methods"]
    pairs = {(a, m) for a in ATTENTION for m in METHODS}
    if len(rows) != len(pairs) or {(r["attention"], r["method"]) for r in rows} != pairs:
        raise ValueError("Summary does not contain all 36 complete method groups")
    for r in rows:
        a, m = r["attention"], r["method"]
        mean = sum(table[a, m, s] for s in range(3)) / 3
        if (not r["complete"] or r["successful_seeds"] != 3 or
                r["test_loss_mean"] != float(mean)):
            raise ValueError(f"Summary mean differs from exact final losses: {a}/{m}")
    metadata = json.loads((LOG.parent / "metadata.json").read_text())
    if metadata["protocol_sha256"] != PROTOCOL:
        raise ValueError("Benchmark metadata has a different protocol fingerprint")
    for name, expected_hash in metadata["protocol"]["source_hashes"].items():
        p = archive_path(name)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen measured source has changed: {name}")
    return table, hashlib.sha256(raw).hexdigest()

