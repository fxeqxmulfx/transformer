"""Modular arithmetic read as two numbers and answered with one: a perceptron's regression.

Source: EXPERIMENT_PLAN.md, "Scalar MLP on mod 97 (2026-10-18)", which asks
why gradient descent does not fit x / y mod p with a perceptron that reads x
and y as two real numbers and answers with one. The proved landscape results
narrow the answer. An MLP's loss has a Lipschitz gradient on bounded sets
only, so they bound the curvature along the run instead: the Hessian has
norm at most L on the segment of every step of gradient descent, and the
step η satisfies ηL < 1 (S). Let the loss be analytic, as it is under GELU,
so that the Łojasiewicz inequality holds at every point
(`Transformer.Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticOnNhd`);
under ReLU it is not C^2. Then almost every start whose orbit satisfies (S)
has an orbit that converges to a critical point that is no strict saddle,
or whose norm tends to infinity
(`Transformer.Landscape.SaddleAvoidance.ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd`).
So a run that does not fit the table ends at a critical point of positive
loss whose Hessian has no negative eigenvalue, such as a spurious local
minimizer or a degenerate saddle, or converges too slowly to fit it within
its budget, or has parameters whose norm grows without bound, or breaks (S)
with a step past the curvature; the regression task measures what tells
them apart, the curvature along the run included. The one-hot
encoding gives every operand value weights of its own, as the free phases
of the phase model do: for |G| ≥ 4 and a step under 1 / (9|G|^2), gradient
descent on them converges from almost every start of energy below
|G|^2 / 8 to the phases of a character, up to periods
(`Transformer.Landscape.PhaseModel.Abelian.ae_exists_tendsto_characterPhase`).
"""

from dataclasses import dataclass

from .benchmarks import Benchmark, is_prime
from .model import Activation, Init, Model, ScaledResidual
from .spec import Spec, require, require_kind

OPERATIONS = ("+", "/")


@dataclass(frozen=True)
class ModularRegression(Benchmark):
    """x op y mod p as one real number, fitted on every x and nonzero y.

    The rows are those of ModularDivision, x in [0, p) and y in [1, p), for
    either operation, so that addition and division read the same inputs;
    the answer c = x op y mod p takes every residue p - 1 times. The target
    is c rescaled to [-1, 1], 2c / (p - 1) - 1; the loss is the mean squared
    error, and a prediction is right when it rounds back to c. Every row
    trains and none is held out: the benchmark asks whether training fits the
    table, not whether it generalizes. Batches walk shuffled epochs with a
    short last batch, so a batch of every row makes each update a full
    gradient step. The rows are fixed, so the data seed is not used.
    """
    prime: int
    operation: str

    def check(self):
        require(is_prime(self.prime), "Modular regression needs a prime modulus")
        require(self.operation in OPERATIONS, "The operation is '+' or '/'")

    @property
    def context(self):
        return 2

    @property
    def observed(self):
        return ("train",)


@dataclass(frozen=True)
class Encoding(Spec, kind=True):
    """How an MLP reads each of its integer inputs, a value v in [0, vocab)."""


@dataclass(frozen=True)
class Scalar(Encoding):
    """The one real number 2v / (vocab - 1) - 1, from -1 at 0 to 1 at vocab - 1."""


@dataclass(frozen=True)
class OneHot(Encoding):
    """The standard basis vector e_v of R^vocab: every value has weights of its own."""


@dataclass(frozen=True)
class MLP(Model):
    """`depth` hidden layers of `width` units on the encoded inputs, then one output number.

    The `context` inputs are encoded and concatenated; each hidden layer is
    an affine map followed by `activation`, and the output an affine map.
    TorchDefault draws each layer's PyTorch default in order, the first layer
    first, its weight before its bias; Normal then redraws every weight
    matrix in the same order.
    """
    width: int
    depth: int
    encoding: Encoding
    activation: Activation
    init: Init
    context: int

    def check(self):
        require(self.width >= 1 and self.depth >= 1 and self.context >= 1,
                "Width, depth and context must be positive")
        require_kind(self.encoding, Encoding, "encoding")
        require_kind(self.activation, Activation, "activation")
        require_kind(self.init, Init, "init")
        require(not isinstance(self.init, ScaledResidual), "An MLP has no residual stream to scale")
