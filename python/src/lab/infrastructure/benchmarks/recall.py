"""Multi-query associative recall, generated as the convex MQAR comparison generated it.

Ports of `experiments/convex_mqar/src/convex_mqar`: `certify.make_example`
(one sequence), `data.load_split` (a split: one NumPy generator per split,
seeded by `SeedSequence([seed, length, split])`, drawing its sequences in
order) and `engine.evaluate` (each chunk's float32 summed cross entropy added
in float64; a query is answered by its most probable token). A row holds the
tokens, then the query positions, then their labels. The model reads the
whole sequence and is scored at the query positions alone.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from ...domain.analysis import milestones
from .samplers import ReseededEpochSampler

SPLITS = ("train", "validation", "test")


def example(rng, length, pairs, vocab, alpha):
    """Distinct keys from the first half of the vocabulary, each followed by its value from the second.

    After the 2 * pairs prefix of key-value pairs, each key recurs once, at
    distinct positions p drawn without replacement with weights p^-alpha, in
    a random order; every other token is a random value token. Returns the
    tokens, the positions of the recurring keys, and the values they recall.
    """
    half = vocab // 2
    keys = rng.choice(half, size=pairs, replace=False)
    values = rng.integers(half, vocab, size=pairs)
    tokens = rng.integers(half, vocab, size=length)
    tokens[:2 * pairs:2], tokens[1:2 * pairs:2] = keys, values
    available = np.arange(2 * pairs, length)
    probability = available.astype(float) ** (-alpha)
    positions = rng.choice(available, size=pairs, replace=False, p=probability / probability.sum())
    order = rng.permutation(pairs)
    tokens[positions] = keys[order]
    return tokens, positions, values[order]


def generate(spec, seed, split):
    """The rows of one split, as one int64 tensor on the host."""
    length, pairs = spec.length, spec.length // 4
    rng = np.random.default_rng(np.random.SeedSequence([seed, length, SPLITS.index(split)]))
    rows = np.empty((getattr(spec, split), length + 2 * pairs), dtype=np.int64)
    for row in rows:
        tokens, positions, labels = example(rng, length, pairs, spec.vocab, spec.alpha)
        row[:length], row[length:length + pairs], row[length + pairs:] = tokens, positions, labels
    return torch.from_numpy(rows)


class RecallTask:
    components = ("loss",)

    def __init__(self, spec, data_seed, device):
        self.spec, self.device, self.vocab = spec, device, spec.vocab
        self.length, self.pairs = spec.length, spec.length // 4
        self.train = generate(spec, data_seed, "train").to(device)
        self.splits = {split: generate(spec, data_seed, split).to(device) for split in ("validation", "test")}

    def sampler(self, batch, seed):
        return ReseededEpochSampler(self.spec.train, batch, seed, self.device)

    def inputs(self, indices):
        """The training rows at `indices`."""
        return self.train.index_select(0, indices.to(self.device))

    def progress(self, seen):
        return {"epochs_seen": seen / self.spec.train}

    def accumulator(self):
        """Evaluation sums: the summed loss and the queries answered correctly."""
        return torch.zeros(2, dtype=torch.float64, device=self.device)

    def forward(self, model, rows):
        """Logits at the query positions, and their labels."""
        tokens, positions, labels = rows.split([self.length, self.pairs, self.pairs], dim=1)
        return model(tokens, positions), labels

    @staticmethod
    def loss(output, labels):
        return F.cross_entropy(output.flatten(0, 1), labels.flatten())

    @staticmethod
    def position_losses(output, labels):
        return RecallTask.loss(output, labels).reshape(1)

    def accumulate(self, model, chunk, sums, static):
        """Add a chunk's float32 summed loss and its correct queries to the float64 `sums`; static either way."""
        output, labels = self.forward(model, chunk)
        loss = F.cross_entropy(output.flatten(0, 1), labels.flatten(), reduction="sum")
        sums.add_(torch.stack([loss.double(), (output.argmax(dim=-1) == labels).sum().double()]))

    def metrics(self, sums, rows):
        """Accuracy over every query and the mean loss per query; None when the loss is not finite."""
        loss, correct = sums
        queries = len(rows) * self.pairs
        return {"accuracy": int(correct) / queries, "loss": loss / queries if math.isfinite(loss) else None,
                "correct": int(correct), "queries": queries}

    @staticmethod
    def observe(model, batch):
        return {}

    @staticmethod
    def inspect(model, batch):
        return {}

    @staticmethod
    def analyze(history):
        """The first observations of the validation accuracy at 50, 75, 90, 95 and 99%."""
        return {"milestones": milestones(history, "validation")}
