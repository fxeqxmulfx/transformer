"""Tasks whose answer the model writes after its input, ended by EOS.

The problem length is the size of the problem: its symbols, bits or digits.
A prompt holds the problem and a separator; the answer follows it, and a
model is scored on the answer it generates from the prompt alone. Every task
reserves `symbols` symbol tokens, used or not, so its vocabulary is that of
the historical suite (`experiments/synthetic_trainers/specs.py`), whose
defaults these are; a task that writes numbers draws them as atomic tokens
up to `number_limit`.

Sources: RASP, arXiv:2106.06981v2, Section 5; RASP-L, arXiv:2310.16028v1,
Sections 4 and 5 and Appendix B (`app:exp-details`). The noncausal outputs
of RASP follow the whole input here.
"""

from dataclasses import dataclass, replace

from .spec import require
from .tasks import Problem, Task

SCRATCHPADS = {"mode": ("none", "counts", "itemized"), "parity": ("none", "running", "ones")}


def check_alphabet(task):
    require(task.symbols >= 2, "Generation needs at least two symbols")
    require(getattr(task, "number_limit", 1) >= 1, "number_limit must be positive")


def check_numbers(task, maximum):
    require(not task.uses_numbers or maximum <= task.number_limit,
            "number_limit must cover every problem length, OOD lengths included")


@dataclass(frozen=True)
class Histogram(Task):
    """For each input symbol, how often it occurs in the input.

    Source: arXiv:2106.06981v2, Section 5, hist_bos, and with `bos` off
    hist_nobos. The answer opens with BOS when the input does.
    """
    symbols: int = 32
    bos: bool = True
    number_limit: int = 512
    generative = uses_numbers = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + (3 if self.bos else 1)

    def check_lengths(self, minimum, maximum):
        check_numbers(self, maximum)


@dataclass(frozen=True)
class DoubleHistogram(Task):
    """For each input symbol, how many distinct symbols occur exactly as often as it.

    Source: arXiv:2106.06981v2, Section 5, hist2, with BOS.
    """
    symbols: int = 32
    number_limit: int = 512
    generative = uses_numbers = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + 3

    def check_lengths(self, minimum, maximum):
        check_numbers(self, maximum)


@dataclass(frozen=True)
class Mode(Task):
    """The most frequent symbol, unique by construction.

    Source: arXiv:2310.16028v1, Section 4 and Appendix B, Mode; a tie is
    broken by moving one occurrence. The paper draws each input from five
    symbols; here from a uniform number of them, two to five and at most the
    length. A `scratchpad` first writes the count of each distinct symbol, as
    a number token, then the symbol: by increasing count, ties by first
    occurrence ("counts", Section 5), or by first occurrence ("itemized").
    The paper's second scratchpad (`app:modescratch`) takes the order of
    "itemized" but writes each symbol before its count.
    """
    symbols: int = 32
    scratchpad: str = "none"
    number_limit: int = 512
    generative = True

    @property
    def uses_numbers(self):
        return self.scratchpad != "none"

    def check(self):
        check_alphabet(self)
        require(self.scratchpad in SCRATCHPADS["mode"], "Mode scratchpad is none, counts or itemized")

    def context(self, length):
        return length + 3 + (2 * min(length, self.symbols) if self.uses_numbers else 0)

    def check_lengths(self, minimum, maximum):
        require(minimum >= 3, "Mode needs three symbols to construct a unique winner")
        check_numbers(self, maximum)


@dataclass(frozen=True)
class MostFrequent(Task):
    """The distinct input symbols by decreasing frequency, ties by first occurrence, padded with BOS.

    Source: arXiv:2106.06981v2, Section 5, most_freq.
    """
    symbols: int = 32
    generative = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + 3


def check_unique(task, maximum):
    require(not task.unique or task.symbols >= maximum,
            "Unique-symbol inputs need as many symbols as the longest problem, OOD lengths included")


@dataclass(frozen=True)
class Copy(Task):
    """The input, written again.

    Source: arXiv:2310.16028v1, Section 4 and Appendix B, Copy: over two
    symbols, drawn from the alphabet for each input (the paper's alphabet
    has two), or with `unique` from the whole alphabet without replacement.
    """
    symbols: int = 32
    unique: bool = False
    generative = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + 2

    def check_lengths(self, minimum, maximum):
        check_unique(self, maximum)


@dataclass(frozen=True)
class Reverse(Task):
    """The input, written backwards.

    Source: arXiv:2106.06981v2, Section 5, reverse, sampled as `Copy` is.
    """
    symbols: int = 32
    unique: bool = False
    generative = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + 2

    def check_lengths(self, minimum, maximum):
        check_unique(self, maximum)


@dataclass(frozen=True)
class Sort(Task):
    """The input symbols in increasing order.

    Source: arXiv:2106.06981v2, Section 5, sort, and arXiv:2310.16028v1,
    Appendix B, Sort, which draws from the alphabet without replacement, as
    `unique` does; otherwise from a uniform number of symbols, two to eight
    and at most the length. The answer omits the BOS that opens RASP's output.
    """
    symbols: int = 32
    unique: bool = False
    generative = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return 2 * length + 2

    def check_lengths(self, minimum, maximum):
        check_unique(self, maximum)


@dataclass(frozen=True)
class Count(Task):
    """Every number from a start to an end, both given; the problem length is how many.

    Source: arXiv:2310.16028v1, Section 4 and Appendix B, Count, with one
    atomic token per number.
    """
    symbols: int = 32
    number_limit: int = 512
    generative = uses_numbers = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return length + 4

    def check_lengths(self, minimum, maximum):
        check_numbers(self, maximum)


@dataclass(frozen=True)
class Addition(Task):
    """The sum of two numbers of up to `length` decimal digits, padded with one leading zero.

    Source: arXiv:2310.16028v1, Section 5 and Appendix B, Addition. The sum
    is written most significant digit first, or with `order` "reverse" last
    first; with `hints` an index token precedes every digit. `carries`
    "standard" draws operands of uniform width; "balanced" draws one carry
    chain of uniform length at a uniform place, which `carry_length` fixes.
    """
    order: str = "forward"
    hints: bool = False
    carries: str = "balanced"
    carry_length: int | None = None
    symbols: int = 32
    number_limit: int = 512
    generative = True

    @property
    def uses_numbers(self):
        return self.hints

    def check(self):
        check_alphabet(self)
        require(self.order in ("forward", "reverse"), "Addition order is forward or reverse")
        require(self.carries in ("standard", "balanced"), "Carries are standard or balanced")
        require(self.carry_length is None or self.carry_length >= 0, "carry_length must be nonnegative")

    def context(self, length):
        return 6 * length + 9 if self.hints else 3 * length + 6

    def check_lengths(self, minimum, maximum):
        require(self.carry_length is None or self.carry_length <= minimum,
                "carry_length must fit every sampled operand length")
        require(not self.hints or maximum + 1 <= self.number_limit, "Indexed addition needs number_limit >= length + 1")
        check_numbers(self, maximum)

    def transfers(self, length, min_length, ood):
        """`hard-carry-length-L`: one carry chain through all L digits, at the training maximum and each OOD length."""
        return {f"hard-carry-length-{n}": Problem(replace(self, carry_length=n), n, None) for n in (length, *ood)}


@dataclass(frozen=True)
class Parity(Task):
    """Whether the input bits hold an odd number of ones.

    Source: arXiv:2310.16028v1, Section 5 and Appendix B, Parity. A
    `scratchpad` writes the running parity, from even, after every bit
    ("running") or after every one ("ones"); `hints` precede every bit, and
    its scratchpad entry, with an index token. The paper's scratchpad is
    "ones" with `hints`.
    """
    scratchpad: str = "none"
    hints: bool = False
    symbols: int = 32
    number_limit: int = 512
    generative = True

    @property
    def uses_numbers(self):
        return self.hints

    def check(self):
        check_alphabet(self)
        require(self.scratchpad in SCRATCHPADS["parity"], "Parity scratchpad is none, running or ones")

    def context(self, length):
        width = 2 if self.hints else 1
        return width * length + 2 + (width * length + 1 if self.scratchpad != "none" else 1)

    def check_lengths(self, minimum, maximum):
        check_numbers(self, maximum)


@dataclass(frozen=True)
class BooleanAnd(Task):
    """Whether every input bit is one; half of the inputs hold a single zero.

    Source: arXiv:2310.16028v1, Appendix B, Boolean-AND. With `shift` the
    zero lies before the last quarter of the input ("early") or in it
    ("late"), and the other region is held out as a position transfer test;
    without it, anywhere.
    """
    shift: bool = True
    region: str = "early"
    symbols: int = 32
    generative = True

    def check(self):
        check_alphabet(self)
        require(self.region in ("early", "late"), "The zero's region is early or late")

    def context(self, length):
        return length + 3

    def check_lengths(self, minimum, maximum):
        require(not self.shift or minimum >= 4, "Position-shift AND needs at least four bits")

    def transfers(self, length, min_length, ood):
        """With `shift`, the zero in the late region: `position_shift` at the training lengths,
        `position-shift-length-L` at each OOD length."""
        if not self.shift:
            return {}
        late = replace(self, region="late")
        return {"position_shift": Problem(late, length, min_length),
                **{f"position-shift-length-{n}": Problem(late, n, None) for n in ood}}


@dataclass(frozen=True)
class RandomLM(Task):
    """A control: independent uniform symbols after a prompt that only states their count.

    Source: arXiv:2505.24832v3, Section 3.2 (not in `papers/`). The answer
    has no algorithm to find; fitting it measures memorization, of
    entropy problem length x log2(symbols) bits per example.
    """
    symbols: int = 32
    number_limit: int = 512
    generative = uses_numbers = True

    def check(self):
        check_alphabet(self)

    def context(self, length):
        return length + 3

    def check_lengths(self, minimum, maximum):
        check_numbers(self, maximum)
