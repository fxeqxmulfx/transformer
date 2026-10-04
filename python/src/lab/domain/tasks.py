"""Algorithmic tasks: the problem families a synthetic benchmark samples.

A task fixes a family and its difficulty controls; its benchmark fixes how
long the problems are. The tasks here label their own input: every position
of a prefix language with the status of its prefix, or every query with its
answer. Tasks that write an answer after their input are in
`lab.domain.generative`. Defaults are those of the historical synthetic suite
(`experiments/synthetic_trainers/specs.py`).
"""

from dataclasses import dataclass
import math
from typing import NamedTuple

from .spec import Spec, require


@dataclass(frozen=True)
class Task(Spec, kind=True):
    """An algorithmic problem family, sampled at the problem lengths its benchmark sets.

    A `generative` task's answer is generated after its prompt; with
    `final_token`, the answer ends with its final answer, the token before
    EOS, which a scratchpad may precede.
    """
    generative = False
    final_token = False
    uses_numbers = False

    def context(self, length):
        """The tokens a model reads for a problem of `length`."""
        return length

    def check_lengths(self, minimum, maximum):
        """Reject problem lengths from `minimum` to `maximum` at which the task cannot be sampled."""

    def transfers(self, length, min_length, ood):
        """Held-out distributions of the task's own, beyond longer problems, by name."""
        return {}


class Problem(NamedTuple):
    """A task at problem lengths `min_length` to `length`; without `min_length`, at `length` alone."""
    task: Task
    length: int
    min_length: int | None

    @property
    def minimum(self):
        return self.length if self.min_length is None else self.min_length


def check_retrieval(task, minimum, required):
    required += task.query_gap
    require(minimum >= required, f"Input needs at least {required} tokens")


@dataclass(frozen=True)
class MQAR(Task):
    """Multi-query associative recall: each query is answered with the value bound to its key.

    Source: the MQAR data procedure of Zoology (arXiv:2312.04927v1, Appendix
    E.1, Procedure 1), as the synthetic trainers adapted it. After BOS the
    input binds `pairs` distinct keys, out of `symbols` key tokens, to values
    out of `symbols` value tokens; the rest of the input is random values,
    among which `queries` of the keys recur, at least `query_gap` tokens after
    the bindings, at positions drawn without replacement with weight
    position^-alpha. A recurring key is supervised with its value. The
    procedure has no BOS and recurs every key: `queries` = `pairs` and
    `query_gap` = 0. So did the convex MQAR comparison
    (`experiments/mqar_sparsemax`), whose keys and values split its
    vocabulary in halves and whose draws were NumPy's.

    `overwrites` extends the procedure: that many of the `pairs` writes
    rebind a key already written, each to a value other than its current
    one, and a query is supervised with the latest value of its key. The
    writes are a uniformly random order of `pairs - overwrites` distinct
    keys, once each, and `overwrites` more drawn uniformly from them; the
    queries are distinct among them. The latest write is the rule that
    settles the ambiguity of the unrestricted setup of Zoology
    (`Transformer.Zoology.unrestricted_mqar_ambiguous`), and the one a
    recency-ordered head keeps only within a window of positions
    (`Transformer.ALM.latest_wins`,
    `Transformer.ALM.past_the_window_rounding_decides`).
    """
    symbols: int = 32
    pairs: int = 8
    queries: int = 4
    query_gap: int = 0
    alpha: float = 0.1
    overwrites: int = 0

    def check(self):
        require(0 <= self.overwrites < self.pairs, "Require 0 <= overwrites < pairs")
        require(1 <= self.queries <= self.pairs - self.overwrites <= self.symbols,
                "Require 1 <= queries <= pairs - overwrites <= symbols")
        require(self.overwrites == 0 or self.symbols >= 2, "A rewrite needs a second value")
        require(self.query_gap >= 0, "query_gap must be nonnegative")
        require(math.isfinite(self.alpha) and self.alpha >= 0, "alpha must be finite and nonnegative")

    def check_lengths(self, minimum, maximum):
        check_retrieval(self, minimum, 1 + 2 * self.pairs + self.queries)


@dataclass(frozen=True)
class Lookup(Task):
    """Composed lookup: each query is answered with the end of a path of `hops` links from it.

    The experimental extension of MQAR in
    `experiments/archive/synthetic_trainers/TASKS.md`, motivated by repeated
    select and aggregate in RASP (arXiv:2106.06981v2, Sections 3.1 and 4).
    After BOS the input lists `pairs` key/value records of one cyclic
    permutation of `pairs` identities, out of `symbols`, then `queries`
    distinct starts; the end of each path is supervised at its start. Every
    link of a path precedes its query, and no path repeats a vertex.
    """
    symbols: int = 32
    pairs: int = 8
    queries: int = 4
    hops: int = 2
    query_gap: int = 0

    def check(self):
        require(1 <= self.queries <= self.pairs <= self.symbols, "Require 1 <= queries <= pairs <= symbols")
        require(self.query_gap >= 0, "query_gap must be nonnegative")
        require(1 <= self.hops < self.pairs, "Lookup requires 1 <= hops < pairs")

    def check_lengths(self, minimum, maximum):
        check_retrieval(self, minimum, 2 + 4 * self.pairs + 2 * self.queries)


def check_neutral(task):
    require(math.isfinite(task.neutral_fraction) and 0 <= task.neutral_fraction < 1,
            "neutral_fraction must be in [0, 1)")
    require(task.max_neutral_gap is None or task.max_neutral_gap >= 0, "max_neutral_gap must be nonnegative")


def check_neutral_lengths(task, minimum, maximum, needed):
    """Every length leaves `needed` active symbols, and its neutral tokens fit the gap bound."""
    for length in range(minimum, maximum + 1):
        active = task.active_size(length)
        require(active >= needed, f"Too few active symbols: need at least {needed}")
        require(task.max_neutral_gap is None or length - 1 - active <= task.max_neutral_gap * (active + 1),
                "Neutral tokens do not fit max_neutral_gap")


def active_size(task, length, even):
    """Symbols after BOS that are not neutral: a `1 - neutral_fraction` share, rounded."""
    size = round((length - 1) * (1 - task.neutral_fraction))
    return size - size % 2 if even else size


@dataclass(frozen=True)
class Dyck(Task):
    """Dyck-1 prefix status: balanced, still open, or invalid, at every position.

    Source: arXiv:2506.16055v3, Example 2.2 (`ex:dyck`) and Appendix A.2;
    the language of the formula `Transformer.CRASP.dyck`. A `neutral_fraction`
    of the input is a neutral letter that leaves the status unchanged, in gaps
    of at most `max_neutral_gap`; the nesting depth stays within `max_balance`.
    """
    neutral_fraction: float = 0.25
    max_neutral_gap: int | None = None
    max_balance: int = 8

    def check(self):
        check_neutral(self)
        require(self.max_balance >= 1, "max_balance must be positive")

    def active_size(self, length):
        return active_size(self, length, even=True)

    def check_lengths(self, minimum, maximum):
        check_neutral_lengths(self, minimum, maximum, 6)


@dataclass(frozen=True)
class AlternatingBlocks(Task):
    """E_k: whether the prefix, its neutral letters deleted, is k alternating blocks of a and b, from a.

    Source: arXiv:2506.16055v3, Appendix F, the language E_k of
    `thm:tlclpos_depth_hierarchy`, here `Transformer.CRASP.altPlusNeutral`,
    with k = `blocks`. A `neutral_fraction` of the input is the neutral
    letter, in gaps of at most `max_neutral_gap`.
    """
    blocks: int = 3
    neutral_fraction: float = 0.25
    max_neutral_gap: int | None = None

    def check(self):
        check_neutral(self)
        require(self.blocks >= 1, "blocks must be positive")

    def active_size(self, length):
        return active_size(self, length, even=False)

    def check_lengths(self, minimum, maximum):
        check_neutral_lengths(self, minimum, maximum, self.blocks + (2 if self.blocks > 1 else 1))


@dataclass(frozen=True)
class TypedDyck(Task):
    """Dyck-i prefix status over `types` kinds of brackets: balanced, still open, or invalid.

    Source: arXiv:2106.06981v2, Section 5, Dyck-i PTF for i = 1, 2; more than
    two types extend the benchmark, up to eight. Neutral letters fill a
    `neutral_fraction` of the input, as for `Dyck`.
    """
    types: int = 2
    max_balance: int = 8
    neutral_fraction: float = 0.25
    max_neutral_gap: int | None = None

    def check(self):
        check_neutral(self)
        require(2 <= self.types <= 8, "Typed Dyck takes between 2 and 8 bracket types")
        require(self.max_balance >= 1, "max_balance must be positive")

    def active_size(self, length):
        return active_size(self, length, even=True)

    def check_lengths(self, minimum, maximum):
        check_neutral_lengths(self, minimum, maximum, 4)


@dataclass(frozen=True)
class CRASP(Task):
    """Whether each prefix satisfies one past-counting formula, drawn once from seed `program`.

    Source: arXiv:2506.16055v3, Section 2.2, the syntax of TL[past-count],
    over the letters a, b and c. `depth` nests counts syntactically; the
    formula drawn is the first, of 64, that is not constant on a bank of
    witness words. Its count depth need not be the least of any equivalent
    formula.
    """
    depth: int = 2
    program: int = 0

    def check(self):
        require(1 <= self.depth <= 5, "C-RASP formulas nest counts 1 to 5 deep")
        require(self.program >= 0, "The program seed is nonnegative")

    def check_lengths(self, minimum, maximum):
        require(minimum >= 2, "C-RASP inputs need a letter after BOS")
