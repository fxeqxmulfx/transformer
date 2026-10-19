"""The browser arithmetic split and continuing short-tail epoch sampler.

Adapted from grokking_abmod/web/own-channel/python/rotate90.py at
f49969393a7482490d45483d8a4fd3046b1adede (MIT; third_party/rotate90/NOTICE).
The p=97 half split and seed streams match that source. Extensions: arbitrary
prime and train fraction, the lab's checkpoint interface, norm and character
geometry observations, and a batch size argument. Evaluation counts optimizer
updates, not browser epochs. The observations do not change training.
"""

import hashlib

import torch
from torch.nn import functional as F

from ..rotary_random import SplitMix64
from ..rotary_geometry import embedding_geometry, multiplicative_ordering


class LabSampler:
    """Shuffle the preceding permutation in place, exactly as the browser does."""
    traced = ("epoch_tail",)

    def __init__(self, size, batch, seed):
        self.batch, self.cursor = batch, 0
        self.random = SplitMix64(seed ^ 0x6261746368657303)
        self.order = list(range(size))
        self.shuffle()

    def shuffle(self):
        self.random.shuffle(self.order)
        self.permutation = torch.tensor(self.order, dtype=torch.long)

    def sizes(self):
        size = len(self.order)
        return sorted({min(self.batch, size), size % self.batch} - {0})

    def next(self):
        size = len(self.order)
        if self.cursor == size:
            self.shuffle()
            self.cursor = 0
        start = self.cursor
        self.cursor = min(size, start + self.batch)
        return [(self.permutation, start, self.cursor - start)], {
            "cursor_before": start, "cursor_after": self.cursor,
            "epoch_tail": self.cursor == size, "epoch_wraps_in_batch": 0}

    def state(self):
        return {"rotary_random_state": self.random.state, "permutation": self.permutation,
                "cursor": self.cursor}

    def restore(self, checkpoint):
        self.random.state = checkpoint["rotary_random_state"]
        self.permutation = checkpoint["permutation"].cpu()
        self.order = self.permutation.tolist()
        self.cursor = checkpoint["cursor"]


def fingerprint(rows):
    return hashlib.sha256(rows.cpu().numpy().astype("<i8").tobytes()).hexdigest()


class LabDivisionTask:
    """One answer-class prediction per equation, with no EOS objective."""
    components = ("answer_loss",)

    def __init__(self, spec, data_seed, device):
        self.spec, self.device, self.data_seed = spec, device, data_seed
        self.vocab = spec.prime
        self.generator, order = multiplicative_ordering(spec.prime)
        self.nonzero_order = torch.tensor(order, dtype=torch.long, device=device)
        inverses = {b: pow(b, -1, spec.prime) for b in range(1, spec.prime)}
        rows = [(a, b, a * inverses[b] % spec.prime)
                for a in range(spec.prime) for b in range(1, spec.prime)]
        if any((q * b - a) % spec.prime for a, b, q in rows):
            raise ValueError("The independent multiplication oracle rejected a division row")
        SplitMix64(data_seed ^ 0x73706C6974730002).shuffle(rows)
        rows = torch.tensor(rows, dtype=torch.long, device=device)
        self.train_count = round(len(rows) * spec.train_fraction)
        self.splits = {"train": rows[:self.train_count], "heldout": rows[self.train_count:]}

    def summary(self):
        return {"prime": self.vocab, "domain_size": self.vocab * (self.vocab - 1),
                "train_examples": self.train_count, "heldout_examples": len(self.splits["heldout"]),
                "train_fraction": self.spec.train_fraction, "data_seed": self.data_seed,
                "vocab_size": self.vocab, "train_fingerprint": fingerprint(self.splits["train"]),
                "heldout_fingerprint": fingerprint(self.splits["heldout"]),
                "split_policy": "rotate90_splitmix64_numerator_denominator_order",
                "character_generator": self.generator,
                "scope": "unseen_operand_pairs_at_one_fixed_prime; no_integer_or_length_OOD"}

    def sampler(self, batch, seed):
        return LabSampler(self.train_count, batch, seed)

    def inputs(self, indices, static=False):
        return self.splits["train"].index_select(0, indices.to(self.device))

    def progress(self, seen):
        return {"epochs_seen": seen / self.train_count}

    def accumulator(self):
        return torch.zeros(2, dtype=torch.float64, device=self.device)

    @staticmethod
    def forward(model, batch, supervised=False):
        return model(batch[:, :2]), batch[:, 2]

    @staticmethod
    def loss(output, targets):
        return F.cross_entropy(output, targets)

    @staticmethod
    def position_losses(output, targets):
        return F.cross_entropy(output, targets).reshape(1)

    @staticmethod
    def accumulate(model, chunk, sums, static):
        logits, targets = LabDivisionTask.forward(model, chunk)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        correct = (logits.argmax(dim=-1) == targets).sum()
        sums.add_(torch.stack((loss.double(), correct.double())))

    @staticmethod
    def metrics(sums, rows):
        loss, correct = sums
        return {"loss": loss / len(rows), "accuracy": correct / len(rows),
                "answer_loss": loss / len(rows), "answer_accuracy": correct / len(rows), "examples": len(rows)}

    def observe(self, model, batch):
        return {"parameter_norm": float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))),
                "embedding_norm": float(model.E.detach().norm()),
                "zero_embedding_norm": float(model.E[0].detach().norm()),
                "embedding_geometry": embedding_geometry(model.E, self.nonzero_order)}

    def inspect(self, model, batch):
        return self.observe(model, batch)

    @staticmethod
    def analyze(history):
        def crossing(split):
            return next((point["step"] for point in history if point[split]["accuracy"] >= 0.99), None)
        return {"first_train99_update": crossing("train"), "first_heldout99_update": crossing("heldout")}
