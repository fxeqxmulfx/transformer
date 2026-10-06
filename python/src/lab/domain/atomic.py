"""Numerical experiments for the bounded, genuine-head convex atomic model.

Source: Transformer.GPTMini.Sparsemax.matchingHeadOutput,
matchingMixtureSample, matchingMixture_minimum_exists and
matchingMixture_gap_certificate, at 83d985f. Numerical heads use independent
shared token Q/K/value tables and the same causal sparsemax. Floating-point
arithmetic, finite storage and approximate pricing are explicit deviations.
"""

from dataclasses import dataclass
import math

from .benchmarks import Benchmark
from .model import Model
from .optimizers import Optimizer
from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class AtomicMatching(Model):
    """Probability mixture of freely selected original Q/K/value heads.

    Source: matchingHeadBox and matchingMixtureDomain at 83d985f. There is
    no score scaling, position embedding, residual, FFN or output map.
    Storage holds at most `heads` selected atoms, not a predetermined bank.
    Constant Q, evenly spaced token keys and constant values initialize one
    active head. Defaults give zero Q/K. `channels=None`
    makes the original values vocabulary logits. Raw AdamW is an explicitly
    nonconvex control; only column training enforces the numerical box.
    """
    context: int
    width: int
    heads: int
    cap: float
    channels: int | None = None
    initial_value: float = 0.0
    initial_query: float = 0.0
    initial_key_spread: float = 0.0
    precision: str = "float32"

    def check(self):
        require(all(type(n) is int and n >= 1 for n in (self.context, self.width, self.heads)),
                "AtomicMatching context, width and head capacity must be positive integers")
        require(math.isfinite(self.cap) and self.cap > 0, "AtomicMatching cap must be finite and positive")
        require(self.channels is None or type(self.channels) is int and self.channels >= 1,
                "AtomicMatching channels must be positive or None")
        require(math.isfinite(self.initial_value) and abs(self.initial_value) <= self.cap,
                "The initial value must lie in the physical head box")
        require(all(math.isfinite(n) and abs(n) <= self.cap
                    for n in (self.initial_query, self.initial_key_spread)),
                "The initial Q/K coordinates must lie in the physical head box")
        require(self.precision in ("float32", "float64"), "AtomicMatching precision is float32 or float64")


@dataclass(frozen=True)
class PairedMatching(AtomicMatching):
    """Encode current and preceding tokens in separate learned key channels.

    Source: pairedHeadLift and pairedHeadOutput in
    src/Transformer/GPTMini/Sparsemax/PairedMatchingHead.lean, a new causal
    encoder for the genuine atomic family following Appendix A.4 of
    arXiv:2211.11052v1. The first half of K comes from the current token;
    the second comes from its preceding token (itself at position zero).
    Q and original values use the current token. All Q/K/value coordinates
    remain freely learned. The model stores token tables, not a V-squared
    pair dictionary. `width` counts both halves, so it must be even.
    Floating point, finite storage and approximate pricing are deviations.
    """

    def check(self):
        super().check()
        require(self.width % 2 == 0, "PairedMatching needs an even width for current and preceding keys")


@dataclass(frozen=True)
class MatchingOrders(Benchmark):
    """The two actual order contexts [0,1] and [1,0], observed at row one.

    Source: matchingOrderTokens and matchingOrderTarget at 5d91bc4.
    The default squared-error sum has optimum zero; any uniform-query head
    has error at least 1/8. Training and selection use the same two points:
    this checks the formal finite example, not unseen-data generalization.
    Other answer targets test positive-error attainment without route labels.
    """
    targets: tuple[float, float] = (1.0, 0.5)

    def check(self):
        require(len(self.targets) == 2 and all(math.isfinite(y) for y in self.targets),
                "MatchingOrders needs two finite answer targets")

    @property
    def context(self):
        return 2

    @property
    def observed(self):
        return ("train", "validation")

    @property
    def selection(self):
        return "validation"


@dataclass(frozen=True)
class MatchingBindings(MatchingOrders):
    """Two valid key/value assignments with the same query and visible multiset.

    Source: pairedBindingTokens and pairedBinding_fit in
    src/Transformer/GPTMini/Sparsemax/PairedMatchingWitness.lean. Contexts
    [0,1,3,2,4,1] and [0,1,4,2,3,1] ask key one's value, encoded by the
    independent scalar values of tokens three and four. Answer targets are
    (0,1), with no supplied route labels. The content-only model's optimum
    squared-error sum is 1/2; a paired head attains zero. Both splits contain
    the same two observations; this is a finite structural check.
    """
    targets: tuple[float, float] = (0.0, 1.0)

    @property
    def context(self):
        return 6


@dataclass(frozen=True)
class Pricing(Spec, kind=True):
    """How a new physical head minimizes the supporting output functional."""


@dataclass(frozen=True)
class OrderPricing(Pricing):
    """Exact global price on the two MatchingOrders contexts, width/cap one.

    Source: matchingHeadOutput_bounds and the two-slot sparsemax formula,
    Eq. (1) of arXiv:1602.02068v2. K=(-1/2,1/2), values=(-1,1) and
    Q=(-sign(g1),-sign(g0)) attain the universal lower bound -sum(abs(g)).
    This analytic oracle is specific to these two observations, not Basis.
    """


@dataclass(frozen=True)
class BindingPricing(Pricing):
    """Exact global price on MatchingBindings with scalar cap-one paired heads.

    Source: pairedBinding_scores and pairedBinding_fit. Q/K's preceding
    channel selects the value immediately after key one. Original values
    for tokens three/four take -sign(g0)/-sign(g1), attaining the universal
    price lower bound -sum(abs(g)). This oracle is specific to these two
    observations; numerical pricing and Basis do not use it.
    """


@dataclass(frozen=True)
class SearchPricing(Pricing):
    """Projected Adam search over free Q/K, with exact conditional values.

    Source: matchingHeadPrice at 83d985f. At each Q/K point, values minimize
    their linear price exactly over the box. Multiple fresh random starts
    search the remaining nonconvex Q/K problem; no global price guarantee
    is asserted. Every randomness source travels in checkpoints.
    """
    restarts: int = 8
    steps: int = 128
    lr: float = 0.1

    def check(self):
        require(all(type(n) is int and n >= 1 for n in (self.restarts, self.steps)),
                "SearchPricing needs positive restart and step counts")
        require(math.isfinite(self.lr) and self.lr > 0, "SearchPricing learning rate must be positive")


@dataclass(frozen=True)
class AtomicColumns(Optimizer):
    """Generate physical atoms, then minimize the convex active-weight problem.

    Source: matchingMixture_criterion_convex and matchingMixture_gap_certificate
    at 5d91bc4; paired mixtures use pairedMixture_criterion_convex. The two
    scalar examples solve their small simplex QPs by active-set enumeration.
    Other criteria use projected weights and line search.
    Finite corrective steps, storage capacity and floating-point errors are
    deviations from an ideal infinite-family convex solver. With SearchPricing
    the found-head gap is not a global certificate; the separate output-box
    bound remains a valid (usually loose) certificate for the current batch.
    """
    pricing: Pricing
    correction_steps: int = 64
    tolerance: float = 1e-10

    def check(self):
        require_kind(self.pricing, Pricing, "pricing")
        require(type(self.correction_steps) is int and self.correction_steps >= 1,
                "AtomicColumns needs positive corrective steps")
        require(math.isfinite(self.tolerance) and self.tolerance > 0,
                "AtomicColumns tolerance must be finite and positive")
