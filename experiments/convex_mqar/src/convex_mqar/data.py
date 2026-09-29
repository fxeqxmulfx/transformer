"""Frozen common data splits; no labels or test statistics enter training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .certify import make_example


def split_seed(seed, length, split):
    return np.random.SeedSequence([seed, length, {"train": 0, "validation": 1, "test": 2}[split]])


def load_split(root, config, length, split):
    count = {"train": config.train_examples, "validation": config.validation_examples,
             "test": config.test_examples}[split]
    spec = {"version": 1, "seed": config.seed, "length": length, "split": split,
            "count": count, "vocab": config.vocab, "alpha": config.alpha,
            "pairs": length // 4, "position_weights": "p^-alpha",
            "filler": "value tokens", "distinct_query_positions": True}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(root) / f"{split}-{identity}"
    metadata = directory / "metadata.json"
    if not metadata.exists():
        directory.mkdir(parents=True, exist_ok=True)
        pairs = spec["pairs"]
        arrays = {"tokens": np.lib.format.open_memmap(directory / "tokens.npy", mode="w+",
                   dtype=np.int32, shape=(count, length)),
                  "positions": np.lib.format.open_memmap(directory / "positions.npy", mode="w+",
                   dtype=np.int32, shape=(count, pairs)),
                  "labels": np.lib.format.open_memmap(directory / "labels.npy", mode="w+",
                   dtype=np.int32, shape=(count, pairs))}
        rng = np.random.default_rng(split_seed(config.seed, length, split))
        for row in range(count):
            tokens, positions, labels = make_example(rng, length, pairs, config.vocab, config.alpha)
            arrays["tokens"][row] = tokens
            arrays["positions"][row] = positions
            arrays["labels"][row] = labels
            if (row + 1) % 10_000 == 0:
                print(json.dumps({"event": "data", "split": split, "length": length,
                                  "generated": row + 1, "total": count}), flush=True)
        for array in arrays.values():
            array.flush()
        metadata.write_text(json.dumps(spec, indent=2) + "\n")
    if json.loads(metadata.read_text()) != spec:
        raise RuntimeError("Data cache does not match the requested configuration")
    tensors = tuple(torch.from_numpy(np.load(directory / f"{name}.npy")).long()
                    for name in ("tokens", "positions", "labels"))
    return tensors, identity


def validate_batch(tokens, positions, labels, vocab):
    """Check the actual causal successor relation on raw sequences."""
    pairs = positions.shape[1]
    keys, values = tokens[:, :2 * pairs:2], tokens[:, 1:2 * pairs:2]
    query_keys = tokens.gather(1, positions)
    matches = query_keys[:, :, None] == keys[:, None, :]
    if not torch.all(matches.sum(-1) == 1):
        raise AssertionError("Every labeled query needs one distinct prior key")
    matched_values = values[:, None, :].expand(-1, pairs, -1).gather(
        2, matches.long().argmax(-1, keepdim=True)).squeeze(-1)
    if not torch.equal(matched_values, labels):
        raise AssertionError("Labels violate raw MQAR")
    if not torch.all(positions >= 2 * pairs):
        raise AssertionError("A labeled query precedes the key-value prefix")
    if not torch.all(keys < vocab // 2) or not torch.all(values >= vocab // 2):
        raise AssertionError("Key/value vocabulary split is invalid")
