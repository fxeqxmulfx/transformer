"""Benchmarks: the task a model is trained and measured on."""

from dataclasses import dataclass
import math

from .spec import Spec, require


@dataclass(frozen=True)
class Benchmark(Spec, kind=True):
    """A task with fixed splits, a training loss and evaluation metrics."""

    @property
    def context(self):
        """Longest input the model reads."""
        raise NotImplementedError

    @property
    def observed(self):
        """The splits evaluated at every observation."""
        raise NotImplementedError

    @property
    def selection(self):
        """The observed split whose metrics select the best observation, if any."""
        return None

    def rank(self, metrics):
        """How an observation of the selection split ranks; the first of the highest rank is the best."""
        return -metrics["loss"]

    @property
    def final(self):
        """The splits evaluated once, on the model of the best observation."""
        return ()

    @property
    def last(self):
        """The splits evaluated once, on the model at the end of training, before the best is restored."""
        return ()


def is_prime(number):
    return number >= 2 and all(number % divisor for divisor in range(2, math.isqrt(number) + 1))


@dataclass(frozen=True)
class ModularDivision(Benchmark):
    """x / y mod p for every x and nonzero y, split once into train and held out.

    Source: Convexifying Transformers (arXiv:2211.11052v1), Section 4, in the
    token format of the openai/grok author code (commit 3d64b1d8): the row
    `<eos> x / y = q <eos>` supervises the answer and the final EOS. The split
    shuffles all p (p - 1) rows with the data seed and trains on the first
    round(fraction * rows).

    Batches walk shuffled epochs of the training rows. With tail "short" an
    epoch ends with its remainder as a smaller batch; with "wrap" a batch fills
    up from the next epoch.
    """
    prime: int
    train_fraction: float
    tail: str = "short"

    def check(self):
        require(is_prime(self.prime), "Division needs a prime modulus")
        require(0 < self.train_fraction < 1, "Train fraction lies strictly between zero and one")
        rows = self.prime * (self.prime - 1)
        require(0 < round(rows * self.train_fraction) < rows, "Both splits must be nonempty")
        require(self.tail in ("short", "wrap"), "Batch tail policy is 'short' or 'wrap'")

    @property
    def context(self):
        return 6

    @property
    def observed(self):
        return ("train", "heldout")


@dataclass(frozen=True)
class TinyShakespeare(Benchmark):
    """Next-character prediction on Tiny Shakespeare.

    Source: the char-rnn corpus (Karpathy, 2015) of 1,115,393 characters, as
    the historical GPTMini text benchmarks read it (`optimizer_benchmark.data`):
    the vocabulary is the sorted set of its characters, and the text splits at
    90% and 95% into train, validation and test. Each update draws `batch`
    windows of `window` characters at uniformly random starts in the training
    text and predicts every next character. Evaluation reads a split as
    consecutive nonoverlapping windows. Validation selects the best
    observation, and the test split is evaluated once, on its model. The
    splits are fixed, so the data seed is not used.
    """
    window: int

    def check(self):
        require(self.window >= 1, "Windows hold at least one character")

    @property
    def context(self):
        return self.window

    @property
    def observed(self):
        return ("validation",)

    @property
    def selection(self):
        return "validation"

    @property
    def final(self):
        return ("test",)


@dataclass(frozen=True)
class AssociativeRecall(Benchmark):
    """Multi-query associative recall (MQAR), as the convex MQAR comparison generated it.

    Source: the MQAR data procedure of Zoology (arXiv:2312.04927v1, Appendix
    E.1, Procedure 1), with the choices it leaves open made by
    `certify.make_example`: a sequence of `length` tokens opens with
    length // 4 adjacent key-value pairs, distinct keys from the first half of
    the `vocab` tokens and values from the second; each key recurs once, at
    distinct later positions p drawn with weights p^-alpha; every other token
    is a random value. The model reads the whole sequence and is scored on the
    value it predicts at each recurring key. Each split is drawn by its own
    generator, seeded by the data seed, the length and the split.

    Batches walk shuffled epochs of the training sequences, each shuffled on
    the training device by a generator seeded with the batch seed plus the
    epoch, and an epoch ends with its remainder as a smaller batch.
    Validation selects the best observation by accuracy, then loss, and the
    test split is evaluated once, on its model. The historical trainer
    observed after each epoch alone; here the initial model is observed too,
    and is the best if no later observation ranks higher.
    """
    length: int
    vocab: int
    alpha: float
    train: int
    validation: int
    test: int

    def check(self):
        require(self.length >= 4, "A sequence holds at least one key-value pair and its query")
        require(self.length // 4 <= self.vocab // 2, "Distinct keys need length / 4 tokens in half the vocabulary")
        require(math.isfinite(self.alpha), "The position exponent must be finite")
        require(min(self.train, self.validation, self.test) >= 1, "Every split must be nonempty")

    @property
    def context(self):
        return self.length

    @property
    def observed(self):
        return ("validation",)

    @property
    def selection(self):
        return "validation"

    def rank(self, metrics):
        return (metrics["accuracy"], -metrics["loss"])

    @property
    def final(self):
        return ("test",)
