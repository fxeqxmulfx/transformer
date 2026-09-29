"""Memory-efficient port of Zoology iclr24 data/associative_recall.py::_mqar.

Keep the RandomState draws and their order, without tiling vocabulary arrays
100,000 times. Zero fillers, distinct values, and even query positions match
the source. Validation at seed+20 is an added split; source test uses seed+10.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import write_json


def generate(vocab, length, pairs, count, seed, alpha):
    if length % 2 or vocab <= length or 4*pairs > length:
        raise ValueError("Invalid Zoology MQAR dimensions")
    rng = np.random.RandomState(seed)
    keys = np.stack([rng.choice(np.arange(1, vocab//2), pairs, replace=False)
                     for _ in range(count)])
    values = np.stack([rng.choice(np.arange(vocab//2, vocab), pairs, replace=False)
                       for _ in range(count)])
    context = 2*pairs
    space = (length-context)//2
    probability = alpha*np.arange(1, space+1)**(alpha-1)
    probability /= probability.sum()
    gaps = np.stack([rng.choice(np.arange(space), pairs, replace=False, p=probability)
                     for _ in range(count)])
    positions = context+2*gaps
    tokens = np.zeros((count, length), dtype=np.int32)
    tokens[:, :context:2], tokens[:, 1:context:2] = keys, values
    np.put_along_axis(tokens, positions, keys, axis=1)
    order = np.argsort(positions, axis=1)
    positions = np.take_along_axis(positions, order, axis=1).astype(np.int32)
    labels = np.take_along_axis(values, order, axis=1).astype(np.int32)
    return tokens, positions, labels


def load_split(root, config, length, split):
    count = {"train": config.train_examples, "validation": config.validation_examples,
             "test": config.test_examples}[split]
    seed = config.data_seed + {"train": 0, "validation": 20, "test": 10}[split]
    spec = {"version": 1, "source_commit": config.source_commit, "split": split,
            "count": count, "seed": seed, "vocab": config.vocab, "length": length,
            "pairs": config.pair_count(length), "alpha": config.alpha,
            "filler": "zero", "values_without_replacement": True,
            "position_weights": "alpha*p^(alpha-1)", "query_positions": "even"}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(root) / f"zoology-{split}-{identity}"
    metadata = directory / "metadata.json"
    if not metadata.exists():
        directory.mkdir(parents=True, exist_ok=True)
        arrays = generate(config.vocab, length, spec["pairs"], count, seed, config.alpha)
        for name, array in zip(("tokens", "positions", "labels"), arrays):
            np.save(directory / f"{name}.npy", array)
        write_json(metadata, spec)
        print(json.dumps({"event": "data", **spec, "identity": identity}), flush=True)
    if json.loads(metadata.read_text()) != spec:
        raise RuntimeError("Zoology cache does not match the requested configuration")
    tensors = tuple(torch.from_numpy(np.load(directory / f"{name}.npy")).long()
                    for name in ("tokens", "positions", "labels"))
    return tensors, identity
