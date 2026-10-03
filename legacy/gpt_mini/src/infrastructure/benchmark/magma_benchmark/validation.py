"""Independent artifact and CUDA checkpoint checks after the experiment."""

from gpt_mini.infrastructure.benchmark.paths import artifact_path as _artifact_path, DATA_FILE as _data_file

import hashlib
import json
import math
from pathlib import Path

import torch

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData, select_learning_rates
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import evaluate, run_id, state_hash
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results, summarize, write_json
from .protocol import baseline, check_compatible, check_pairing, source_hashes


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    protocol = metadata["protocol"]
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata["protocol_sha256"]:
        raise ValueError("Metadata protocol fingerprint mismatch")
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("Measured sources have changed")
    old_rows, old_metadata, provenance = baseline()
    if provenance != protocol["baseline"]:
        raise ValueError("Baseline provenance mismatch")
    check_compatible(protocol, old_metadata["protocol"])
    rows = read_results(directory / "runs.jsonl")
    check_pairing(rows, old_rows)
    selected = json.loads((directory / "selected_rates.json").read_text())
    names = [m["name"] for m in protocol["methods"]]
    computed = {a: select_learning_rates([r for r in rows if r["attention"] == a], names)
                for a in protocol["attention"]}
    if selected != computed:
        raise ValueError("Rates were not selected solely by screening validation loss")
    expected = {run_id(a, m["name"], lr, 0, "screen", protocol["screen_steps"])
                for a in protocol["attention"] for m in protocol["methods"] for lr in m["rates"]}
    expected |= {run_id(a, name, selected[a][name], s, "final", protocol["final_steps"])
                 for a in protocol["attention"] for name in names if selected[a][name] is not None
                 for s in protocol["seeds"]}
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise ValueError("Missing, unexpected, or duplicate run identifiers")
    mask_streams = {}
    for row in rows:
        if row["protocol_sha256"] != fingerprint:
            raise ValueError("Raw row has a different protocol")
        if row["status"] == "ok" and row["completed_steps"] != row["steps"]:
            raise ValueError("Successful row has an incomplete update budget")
        if row["phase"] == "screen" and row["test_loss"] is not None:
            raise ValueError("Screening evaluated test loss")
        if row["status"] == "ok":
            fields = ["validation_loss"] + (["test_loss"] if row["phase"] == "final" else [])
            if not all(math.isfinite(row[field]) for field in fields):
                raise ValueError("Successful run has a nonfinite held-out loss")
        if row["magma"] is not None:
            blocks = row["magma"]["blocks"]
            if len(blocks) != 8 or row["magma"]["mask_draws"] != 8 * row["completed_steps"]:
                raise ValueError("Unexpected mask block/update count")
            for b in blocks.values():
                if (b["draws"] != row["completed_steps"] or not 0 <= b["kept"] <= b["draws"]
                        or not 0.3775 <= b["final_damping"] <= 0.6225):
                    raise ValueError("Invalid mask count or damping invariant")
            key = (row["phase"], row["seed"], row["completed_steps"])
            counts = {name: b["kept"] for name, b in blocks.items()}
            if key in mask_streams and mask_streams[key] != counts:
                raise ValueError("Mask stream is not paired across Magma methods and attentions")
            mask_streams[key] = counts
    summary = json.loads((directory / "summary.json").read_text())["methods"]
    if summarize(rows, protocol["seeds"]) != summary:
        raise ValueError("Summary differs from raw measurements")
    torch.set_num_threads(4)
    data = TextData.load(_data_file, "cuda")
    if data.sha256 != protocol["data_sha256"]:
        raise ValueError("Dataset fingerprint changed")
    checkpoints = []
    for row in rows:
        if row["phase"] != "final" or row["seed"] != 0 or row["status"] != "ok":
            continue
        path = _artifact_path(row["checkpoint"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint fingerprint changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["optimizer_steps"] != row["steps"] or payload["magma"] != row["magma"]:
            raise ValueError("Checkpoint optimizer state/diagnostics mismatch")
        model = make_model(Config(**protocol["config"]), row["attention"], 0, "cuda")
        model.load_state_dict(payload["model"])
        if state_hash(model) != row["final_sha256"]:
            raise ValueError("Checkpoint final model hash mismatch")
        loss = evaluate(model, data.test, protocol["config"]["max_seq_len"], protocol["batch"])
        if abs(loss - row["test_loss"]) > 1e-9:
            raise ValueError("Reloaded CUDA checkpoint does not reproduce test loss")
        checkpoints.append({"method": row["method"], "attention": row["attention"],
                            "test_loss": loss, "checkpoint_sha256": digest})
        del model, payload
        torch.cuda.empty_cache()
    result = {"unique_runs": len(rows), "screen_runs": sum(r["phase"] == "screen" for r in rows),
              "final_runs": sum(r["phase"] == "final" for r in rows),
              "failed_runs": sum(r["status"] != "ok" for r in rows),
              "complete_method_groups": sum(r["complete"] for r in summary),
              "paired_initialization_and_batches": True, "paired_mask_streams": True,
              "validation_only_selection": True, "frozen_baseline_unchanged": True,
              "raw_sha256": hashlib.sha256((directory / "runs.jsonl").read_bytes()).hexdigest(),
              "checkpoint_checks": checkpoints}
    write_json(directory / "validation.json", result)
    return result
