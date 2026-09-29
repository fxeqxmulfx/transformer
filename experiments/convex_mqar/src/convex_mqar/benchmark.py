"""Common-test comparison; hyperparameters are selected only on validation."""

import fcntl
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .certify import train_metric
from .convex import ConvexRecall
from .data import load_split, validate_batch
from .engine import evaluate, train_run, write_json
from .rope import RopeTransformer


def run(config, output, data_root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already using this run directory") from None
        return run_locked(config, output, data_root)


def run_locked(config, output, data_root):
    torch.set_num_threads(6)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this benchmark does not silently switch to CPU")
    metadata_path = output / "config.json"
    serialized_config = json.loads(json.dumps(config.to_dict()))
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != serialized_config:
        raise RuntimeError("The output directory belongs to a different configuration")
    write_json(metadata_path, serialized_config)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "numpy": np.__version__,
                   "device": torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
                   "precision": config.precision, "rope_base": config.rope_base}
    source_files = ("rope.py", "engine.py", "data.py", "convex.py", "config.py", "certify.py")
    environment["source_sha256"] = hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes() for name in source_files)).hexdigest()
    write_json(output / "environment.json", environment)
    print(json.dumps({"event": "start", "config": serialized_config,
                      "environment": environment}), flush=True)
    calibration_started = time.perf_counter()
    bits = (config.vocab - 1).bit_length()
    metric, calibration = train_metric(bits, np.random.default_rng(config.seed))
    calibration["seconds"] = time.perf_counter() - calibration_started
    convex = ConvexRecall(config.vocab, metric).to(config.device)
    report = {"config": serialized_config, "environment": environment,
              "convex_training": calibration, "results": [],
              "scope": "Same MQAR test; different inductive biases and matching calibration",
              "selection": "Validation accuracy then validation loss; test never selects a run"}
    for length in config.lengths:
        data = {}
        identities = {}
        for split in ("train", "validation", "test"):
            tensors, identities[split] = load_split(data_root, config, length, split)
            for start in range(0, len(tensors[0]), 1024):
                validate_batch(*(tensor[start:start + 1024] for tensor in tensors), config.vocab)
            data[split] = tuple(tensor.to(config.device) for tensor in tensors)
        if len(set(identities.values())) != 3:
            raise AssertionError("Data splits are not distinct")
        for width in config.widths:
            candidates = []
            for learning_rate in config.learning_rates:
                directory = output / f"n{length}-d{width}-lr{learning_rate:.8g}"
                result, checkpoint = train_run(config, length, width, learning_rate,
                                              data["train"], data["validation"],
                                              directory, output / "status.json")
                candidates.append((result, checkpoint))
            selected, checkpoint = max(candidates, key=lambda item: (
                item[0]["validation"]["accuracy"], -item[0]["validation"]["loss"]))
            model = RopeTransformer(config.vocab, width, config.layers, config.heads,
                                    config.mlp_ratio, config.rope_base).to(config.device)
            model.load_state_dict(torch.load(checkpoint, map_location=config.device,
                                            weights_only=True))
            batch_size = config.batch_size(length, width)
            rope_test = evaluate(model, data["test"], batch_size, config)
            convex_test = evaluate(convex, data["test"], batch_size, config, convex=True)
            row = {"length": length, "pairs": length // 4, "width": width,
                   "splits": identities, "selected_run": selected,
                   "rope": rope_test, "convex": convex_test,
                   "accuracy_difference": convex_test["accuracy"] - rope_test["accuracy"],
                   "convex_metric_parameters": bits}
            report["results"].append(row)
            write_json(output / "comparison.json", report)
            print(json.dumps({"event": "comparison", **row}), flush=True)
            del model
        del data
        if config.device == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "status.json", {"event": "complete", "time_unix": time.time()})
    return report
