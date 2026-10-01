"""Preserve both earlier experiments and pair the longer training streams."""

import hashlib
import json
from pathlib import Path

from experiments.magma_benchmark.protocol import baseline, source_hashes as previous_sources
from experiments.optimizer_benchmark.storage import read_results


MAGMA = Path("experiments/magma_benchmark/results/rtx3050")
MAGMA_RAW = "56ae93f2305935ae5ddc65ff53c0ab657fb5508bf096b4299584809b5d75ee18"


def previous_runs():
    old_rows, old_metadata, _ = baseline()
    metadata = json.loads((MAGMA / "metadata.json").read_text())
    if hashlib.sha256((MAGMA / "runs.jsonl").read_bytes()).hexdigest() != MAGMA_RAW:
        raise ValueError("Frozen Magma raw measurements changed")
    if metadata["protocol"]["source_hashes"] != previous_sources():
        raise ValueError("Frozen Magma sources changed")
    return old_rows + read_results(MAGMA / "runs.jsonl"), old_metadata


def source_hashes():
    result = previous_sources()
    for path in sorted(Path("experiments/patience_benchmark").rglob("*.py")):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def prior_fingerprints():
    names = ("metadata.json", "runs.jsonl", "selected_rates.json")
    dirs = (Path("experiments/optimizer_benchmark/results/rtx3050"), MAGMA)
    return {str(d / name): hashlib.sha256((d / name).read_bytes()).hexdigest()
            for d in dirs for name in names}


def check_initial(row, earlier):
    reference = next(r for r in earlier if r["phase"] == "final" and r["seed"] == row["seed"])
    if row["initial_sha256"] != reference["initial_sha256"]:
        raise ValueError("Longer run uses unpaired initial parameters")
