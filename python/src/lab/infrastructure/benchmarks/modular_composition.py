"""Final-answer sequence tasks; see domain.modular_composition for scope."""

import torch
from torch.nn import functional as F

from ...domain.modular_composition import ModularPairs
from ..rotary_random import SplitMix64
from .rotary import LabSampler, fingerprint
from .power_division import PowerDivisionTask


def chain_answer(inputs, prime):
    """Compute labels using bounded Python integer arithmetic at every step."""
    value = inputs[0]
    for index in range(1, len(inputs), 2):
        value = (value**3 + inputs[index] * value + inputs[index + 1]) % prime
    return value


def unique_inputs(prime, length, count, random):
    """Rejection sampling without replacement, or shuffle a small full domain."""
    domain = prime**length
    if 2 * count > domain:
        codes = list(range(domain))
        random.shuffle(codes)
        rows = []
        for code in codes[:count]:
            digits = []
            for _ in range(length):
                code, digit = divmod(code, prime)
                digits.append(digit)
            rows.append(tuple(digits))
        return rows
    rows, seen = [], set()
    while len(rows) < count:
        row = tuple(random.index(prime) for _ in range(length))
        if row not in seen:
            rows.append(row)
            seen.add(row)
    return rows


class ModularCompositionTask:
    components = ("answer_loss",)

    def __init__(self, spec, data_seed, device):
        self.spec, self.data_seed, self.device, self.vocab = spec, data_seed, device, spec.prime
        random = SplitMix64(data_seed ^ 0x73706C6974730002)
        if isinstance(spec, ModularPairs):
            rows = []
            for a in range(spec.prime):
                for b in range(spec.operation == "division", spec.prime):
                    label = ((a**3 + a * b) % spec.prime if spec.operation == "mixed" else
                             pow(a * pow(b, -1, spec.prime) % spec.prime, spec.exponent, spec.prime))
                    rows.append((a, b, label))
            random.shuffle(rows)
            cut = round(len(rows) * spec.train_fraction)
            splits = {"train": rows[:cut], "heldout": rows[cut:]}
        else:
            inputs = unique_inputs(spec.prime, 1 + 2 * spec.steps,
                                   spec.train_examples + spec.heldout_examples, random)
            rows = [(*row, chain_answer(row, spec.prime)) for row in inputs]
            longer = unique_inputs(spec.prime, 1 + 4 * spec.steps, spec.longer_examples,
                                   SplitMix64(data_seed ^ 0x6C6F6E6765720001))
            splits = {"train": rows[:spec.train_examples], "heldout": rows[spec.train_examples:],
                      "longer": [(*row, chain_answer(row, spec.prime)) for row in longer]}
        self.splits = {name: torch.tensor(rows, dtype=torch.long, device=device).reshape(
            len(rows), len(rows[0]) if rows else 3) for name, rows in splits.items()}
        self.train_count = len(self.splits["train"])
        self.metadata = self.summary()

    def summary(self):
        details = {"prime": self.vocab, "data_seed": self.data_seed, "vocab_size": self.vocab,
                   "split_policy": "unique_inputs_splitmix64_no_replacement",
                   "supervision": "final_answer_only; no target or intermediates in model input"}
        if isinstance(self.spec, ModularPairs):
            details.update(operation=self.spec.operation, exponent=self.spec.exponent,
                           domain_size=self.vocab * (self.vocab - (self.spec.operation == "division")),
                           scope="full_table_fitting_only" if self.spec.train_fraction == 1 else
                           "unseen_pairs_at_fixed_prime")
        else:
            details.update(operation="chain", steps=self.spec.steps,
                           domain_size=self.vocab ** (1 + 2 * self.spec.steps),
                           full_domain_copy_last_probability=(2 * self.vocab - 1) / self.vocab**2,
                           scope="unseen_inputs_and_final_only_double_length; fixed_prime")
        for name, rows in self.splits.items():
            details[name] = {"examples": len(rows), "input_length": rows.shape[1] - 1,
                             "fingerprint": fingerprint(rows),
                             "input_fingerprint": fingerprint(rows[:, :-1]),
                             "copy_last_accuracy": float((rows[:, -2] == rows[:, -1]).double().mean())
                             if len(rows) else None,
                             "majority_accuracy": float(torch.bincount(rows[:, -1], minlength=self.vocab).max())
                             / len(rows) if len(rows) else None}
        return details

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
        return model(batch[:, :-1])[:, -1, :], batch[:, -1]

    @staticmethod
    def loss(output, targets):
        return F.cross_entropy(output, targets)

    @staticmethod
    def position_losses(output, targets):
        return F.cross_entropy(output, targets).reshape(1)

    @staticmethod
    def accumulate(model, chunk, sums, static):
        output, targets = ModularCompositionTask.forward(model, chunk)
        loss = F.cross_entropy(output, targets, reduction="sum")
        correct = (output.argmax(dim=-1) == targets).sum()
        sums.add_(torch.stack((loss.double(), correct.double())))

    @staticmethod
    def metrics(sums, rows):
        loss, correct = sums
        return {"loss": loss / len(rows), "accuracy": correct / len(rows),
                "answer_loss": loss / len(rows), "answer_accuracy": correct / len(rows), "examples": len(rows)}

    def observe(self, model, batch):
        return {"parameter_norm": float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))),
                "parameters": sum(p.numel() for p in model.parameters()), "task": self.metadata}

    inspect = observe
    analyze = staticmethod(PowerDivisionTask.analyze)
