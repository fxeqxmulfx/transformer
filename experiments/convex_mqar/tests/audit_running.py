"""Read-only audit of the active run's source, cached data, and CPU checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from convex_mqar.config import Config
import convex_mqar.benchmark as benchmark


def audit_data(directory, spec):
    n, pairs, count = spec["length"], spec["pairs"], spec["count"]
    half = spec["vocab"] // 2
    arrays = [np.load(directory / f"{name}.npy", mmap_mode="r")
              for name in ("tokens", "positions", "labels")]
    for array, shape in zip(arrays, ((count, n), (count, pairs), (count, pairs))):
        assert array.shape == shape and array.dtype == np.int32, directory
    for start in range(0, count, 1024):
        raw, queries, answers = (a[start:start+1024] for a in arrays)
        keys, values = raw[:, :2*pairs:2], raw[:, 1:2*pairs:2]
        assert np.all((raw >= 0) & (raw < spec["vocab"]))
        assert np.all(keys < half) and np.all(values >= half)
        assert np.all(np.diff(np.sort(keys, axis=1), axis=1) > 0)
        assert np.all((queries >= 2*pairs) & (queries < n))
        assert np.all(np.diff(np.sort(queries, axis=1), axis=1) > 0)
        query_keys = np.take_along_axis(raw, queries, axis=1)
        np.testing.assert_array_equal(np.sort(query_keys, axis=1), np.sort(keys, axis=1))
        for row, row_keys, row_values, qpos, expected in zip(raw, keys, values, queries, answers):
            dictionary = dict(zip(row_keys.tolist(), row_values.tolist()))
            assert [dictionary[int(row[p])] for p in qpos] == expected.tolist()
            filler = np.ones(n, dtype=bool)
            filler[:2*pairs], filler[qpos] = False, False
            assert np.all(row[filler] >= half)
    return {"directory": directory.name, "split": spec["split"], "length": n,
            "sequences": count, "queries": count * pairs, "raw_labels": "passed",
            "unique_keys_and_queries": "passed", "causal_prefix_and_fillers": "passed"}


def audit_checkpoint(path, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    epoch = state["next_epoch"]
    assert 1 <= epoch <= config.epochs
    length = int(path.parent.name.split("-")[0][1:])
    width = int(path.parent.name.split("-")[1][1:])
    expected_steps = epoch * math.ceil(config.train_examples / config.batch_size(length, width))
    for name, tensor in state["model"].items():
        assert torch.isfinite(tensor).all(), (path, name)
    optimizer = state["optimizer"]
    parameters = [index for group in optimizer["param_groups"] for index in group["params"]]
    assert len(parameters) == len(set(parameters)) == len(optimizer["state"])
    for fields in optimizer["state"].values():
        assert fields["step"].item() == expected_steps
        assert all(torch.isfinite(tensor).all() for tensor in fields.values())
    return {"run": path.parent.name, "completed_epoch_snapshot": epoch,
            "optimizer_steps": expected_steps, "finite_model_and_optimizer": "passed"}


def audit(run, data):
    config_json = json.loads((run / "config.json").read_text())
    environment = json.loads((run / "environment.json").read_text())
    config = Config(**config_json)
    names = ("rope.py", "engine.py", "data.py", "convex.py", "config.py", "certify.py")
    source_hash = hashlib.sha256(b"".join(Path(benchmark.__file__).with_name(name).read_bytes()
                                         for name in names)).hexdigest()
    assert source_hash == environment["source_sha256"], "Benchmark source changed during the run"
    datasets = []
    counts = {"train": config.train_examples, "validation": config.validation_examples,
              "test": config.test_examples}
    for metadata in sorted(data.glob("*/metadata.json")):
        spec = json.loads(metadata.read_text())
        if (spec["seed"], spec["vocab"], spec["alpha"]) != (config.seed, config.vocab, config.alpha):
            continue
        if spec["length"] not in config.lengths or spec["count"] != counts.get(spec["split"]):
            continue
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        assert metadata.parent.name == f"{spec['split']}-{identity}"
        datasets.append(audit_data(metadata.parent, spec))
    assert datasets, "No completed datasets found"
    checkpoints = [audit_checkpoint(path, config) for path in sorted(run.glob("*/current.pt"))]
    status = json.loads((run / "status.json").read_text())
    return {"time_unix": time.time(), "device": "cpu", "source_sha256": source_hash,
            "fingerprinted_files": list(names), "source_unchanged": True,
            "datasets": datasets, "sequences_checked": sum(d["sequences"] for d in datasets),
            "queries_checked": sum(d["queries"] for d in datasets), "checkpoints": checkpoints,
            "run_status_snapshot": status,
            "scope": "Read-only snapshot; no CUDA kernels, writes to the run, or stop signals"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
