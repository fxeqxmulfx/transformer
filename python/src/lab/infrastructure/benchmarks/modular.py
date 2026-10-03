"""Prime-field division in the openai/grok token format.

Source: Power et al.'s author code, openai/grok at commit
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, grok/data.py, as ported in
`experiments/synthetic_trainers/paper_reproduction/modular_data.py`; the rows
and the split are identical. The reference format supervises the answer and
the final EOS, with EOS also opening the prompt. An independent inverse
oracle checks every generated row.
"""

from dataclasses import dataclass
import hashlib
import itertools

import numpy as np
import torch
from torch.nn import functional as F

from ...domain.analysis import curve_witness, transition
from ...domain.benchmarks import is_prime
from .samplers import EpochSampler

OPERATORS = ("+", "-", "*", "/", "**2+", "**3+", "x**2+y**2_mod_97",
             "x**2+y**2+x*y_mod_97", "x**2+y**2+x*y+x_mod_97", "x**3+x*y_mod_97",
             "x**3+x*y**2+y_mod_97", "(x._value//y)if(y._value%2==1)else(x-y)_mod_97",
             "s5", "s5conj", "s5aba", "+*", "+-", "sort", "reverse", "copy")


def vocabulary(prime):
    return ("<|eos|>", "=", *sorted(OPERATORS), *(str(number) for number in range(prime)),
            *("".join(map(str, word)) for word in itertools.permutations(range(5))))


def answer(numerator, denominator, prime):
    if not is_prime(prime) or not 0 <= numerator < prime or not 1 <= denominator < prime:
        raise ValueError("Division requires canonical values in a prime field and a nonzero denominator")
    return numerator * pow(denominator, -1, prime) % prime


def complete_rows(prime):
    tokens = vocabulary(prime)
    index = {token: position for position, token in enumerate(tokens)}
    rows = []
    for quotient in range(prime):
        for denominator in range(1, prime):
            numerator = denominator * quotient % prime
            if answer(numerator, denominator, prime) != quotient:
                raise ValueError("Independent division oracle rejected the generated answer")
            rows.append((0, index[str(numerator)], index["/"], index[str(denominator)],
                         index["="], index[str(quotient)], 0))
    return np.array(rows, dtype=np.int64)


def fingerprint(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True)
class Corpus:
    prime: int
    train: np.ndarray
    heldout: np.ndarray
    tokens: tuple[str, ...]
    data_seed: int

    def summary(self):
        return {"prime": self.prime, "domain_size": self.prime * (self.prime - 1),
                "train_examples": len(self.train), "heldout_examples": len(self.heldout),
                "train_fraction": len(self.train) / (self.prime * (self.prime - 1)),
                "data_seed": self.data_seed, "vocab_size": len(self.tokens),
                "train_fingerprint": fingerprint(self.train), "heldout_fingerprint": fingerprint(self.heldout),
                "split_policy": "exhaustive_disjoint_two_way_author_protocol",
                "prompt_tokens": 5, "supervised_targets": 2,
                "scope": "all_unseen_operand_pairs_in_fixed_prime_field; no_length_OOD"}


def make_corpus(spec, data_seed):
    """Shuffle every row with the author's legacy NumPy generator, then split."""
    rows = complete_rows(spec.prime)
    np.random.RandomState(data_seed).shuffle(rows)
    count = round(len(rows) * spec.train_fraction)
    return Corpus(spec.prime, rows[:count], rows[count:], vocabulary(spec.prime), data_seed)


class ModularTask:
    """Training and evaluation on the division corpus, as `paper_reproduction.grokking` did them.

    A row is EOS, x, '/', y, '=', answer, EOS. The model reads the first six
    tokens; its outputs at positions 4 and 5 predict the answer and the final
    EOS, the two supervised targets.
    """
    components = ("answer_loss", "EOS_loss")

    def __init__(self, spec, data_seed, device):
        self.spec, self.device = spec, device
        self.corpus = make_corpus(spec, data_seed)
        self.vocab = len(self.corpus.tokens)
        self.splits = {"train": torch.tensor(self.corpus.train, dtype=torch.long, device=device),
                       "heldout": torch.tensor(self.corpus.heldout, dtype=torch.long, device=device)}

    def summary(self):
        return self.corpus.summary()

    def sampler(self, batch, seed):
        return EpochSampler(len(self.corpus.train), batch, self.spec.tail == "wrap", seed)

    def inputs(self, indices):
        """The training rows at `indices`."""
        return self.splits["train"].index_select(0, indices.to(self.device))

    def progress(self, seen):
        return {"epochs_seen": seen / len(self.corpus.train)}

    def accumulator(self):
        """Evaluation sums: three match counts and three summed losses."""
        return torch.zeros(6, dtype=torch.float64, device=self.device)

    @staticmethod
    def forward(model, batch):
        """Logits at the supervised positions, and their targets."""
        return model(batch[:, :-1])[:, 4:, :], batch[:, 5:]

    @staticmethod
    def loss(output, targets):
        return F.cross_entropy(output.reshape(-1, output.shape[-1]), targets.reshape(-1))

    @staticmethod
    def position_losses(output, targets):
        """The mean loss at each supervised position: answer, then EOS."""
        losses = F.cross_entropy(output.reshape(-1, output.shape[-1]), targets.reshape(-1), reduction="none")
        return losses.reshape(-1, targets.shape[-1]).mean(dim=0)

    @staticmethod
    def accumulate(model, chunk, sums):
        """Add a chunk's match counts and summed losses to the float64 `sums`.

        Each chunk's float32 loss sums are added in float64, as the historical
        evaluation added each `.item()` to a Python float, so the totals are
        bit-identical however the chunks are issued.
        """
        output, target = ModularTask.forward(model, chunk)
        predicted = output.argmax(dim=-1)
        counts = torch.stack([(predicted == target).all(dim=1).sum(), (predicted[:, 0] == target[:, 0]).sum(),
                              (predicted[:, 1] == target[:, 1]).sum()])
        losses = torch.stack([
            F.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1), reduction="sum"),
            F.cross_entropy(output[:, 0], target[:, 0], reduction="sum"),
            F.cross_entropy(output[:, 1], target[:, 1], reduction="sum")])
        sums.add_(torch.cat([counts.double(), losses.double()]))

    @staticmethod
    def metrics(sums, examples):
        correct, answers, stops, loss, answer_loss, stop_loss = sums
        return {"accuracy": correct / examples, "answer_accuracy": answers / examples,
                "EOS_accuracy": stops / examples, "loss": loss / (2 * examples),
                "answer_loss": answer_loss / examples, "EOS_loss": stop_loss / examples, "examples": examples}

    @staticmethod
    def analyze(history):
        """The grokking transition and double-descent witnesses of the held-out curves."""
        return {"transition": transition(history),
                "epoch_loss_witness": curve_witness([(point["step"], point["heldout"]["loss"])
                                                     for point in history], .02),
                "epoch_error_witness": curve_witness([(point["step"], 1 - point["heldout"]["accuracy"])
                                                      for point in history], .02)}
