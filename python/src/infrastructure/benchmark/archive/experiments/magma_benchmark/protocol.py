"""Validate the immutable old experiment and pairing before merging results."""

import hashlib
import json
from pathlib import Path

from experiments.optimizer_benchmark.storage import read_results
from scripts.optimizer_benchmark_data import extract


BASELINE = Path("experiments/optimizer_benchmark/results/rtx3050")
FROZEN_RAW_SHA256 = "01b42cb9e0e42c2ac04173d9c2c723eb75c2bf8cd4e7c9a7654ab19fb74f6f47"
SHARED_FIELDS = ("config", "batch", "screen_steps", "final_steps", "seeds", "attention",
                 "data_sha256", "data_boundaries", "characters", "initialization", "dtype",
                 "tf32", "deterministic_algorithms")


def baseline():
    _, raw_hash = extract()
    if raw_hash != FROZEN_RAW_SHA256:
        raise ValueError("The frozen baseline raw log has changed")
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    protocol_hash = hashlib.sha256(json.dumps(metadata["protocol"], sort_keys=True).encode()).hexdigest()
    if protocol_hash != metadata["protocol_sha256"]:
        raise ValueError("The baseline metadata does not match its protocol fingerprint")
    provenance = {"directory": str(BASELINE), "raw_sha256": raw_hash,
                  "protocol_sha256": protocol_hash,
                  "metadata_sha256": hashlib.sha256((BASELINE / "metadata.json").read_bytes()).hexdigest()}
    return read_results(BASELINE / "runs.jsonl"), metadata, provenance


def check_compatible(new, old):
    for field in SHARED_FIELDS:
        if new[field] != old[field]:
            raise ValueError(f"Cannot compare incompatible {field}")


def check_pairing(new_rows, old_rows):
    references = {(r["phase"], r["seed"]): r for r in old_rows}
    for row in new_rows:
        if row["phase"] not in {"screen", "final"}:
            continue
        old = references[row["phase"], row["seed"]]
        for field in ("initial_sha256", "batch_sha256", "steps", "tokens_seen"):
            if row[field] != old[field]:
                raise ValueError(f"Unpaired {row['id']}: {field}")


def source_hashes():
    paths = sorted(Path("experiments/optimizer_benchmark").rglob("*.py"))
    paths += sorted(Path("experiments/magma_benchmark").rglob("*.py"))
    paths += [Path("experiments/gpt_mini.py"), Path("experiments/test_optimizer_parameter_groups.py"),
              Path("scripts/optimizer_benchmark_data.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
