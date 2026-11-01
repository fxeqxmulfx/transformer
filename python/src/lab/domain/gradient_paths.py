"""Known-solution witnesses for residual amplitude and gradient transport.

Sources: Transformer.GPTMini.rmsNormEps / rmsNorm_pos_homog and
Transformer.Residual.residualRun. Equations and finite-epsilon optima are
specified in experiments/residual_gradient_paths/README.md, Eqs. 1–4.
These supplied-feature regressions are not token-attention or grokking tasks.
"""

from dataclasses import dataclass
import math

from .benchmarks import Benchmark
from .model import Model
from .spec import require


@dataclass(frozen=True)
class BranchGates(Model):
    """Learn 2*sigmoid(s) before/after parameter-free branch RMS, s initially 0.

    Inputs contain an identity vector followed by independent unit-RMS
    Rademacher branch vectors. All parameters are branch logits, not features.
    Epsilon is explicit and defaults to the source FP32 RMSNorm epsilon.
    """
    width: int = 32
    depth: int = 1
    placement: str = "after"
    scale: float = 1.0
    eps: float = 2 ** -23

    def check(self):
        require(self.width >= 32 and self.depth >= 1, "Positive branches and width at least 32")
        require(self.placement in ("before", "after"), "Gate placement is before or after RMS")
        require(math.isfinite(self.scale) and self.scale > 0, "Branch scale is finite and positive")
        require(math.isfinite(self.eps) and self.eps > 0, "RMS epsilon is finite and positive")
        require(self.scale / math.sqrt(self.scale ** 2 + self.eps) > .75,
                "Both bounded parameterizations must represent every teacher coefficient")

    @property
    def context(self):
        return self.depth + 1


@dataclass(frozen=True)
class LinearTransport(Model):
    """Learn a scalar before D fixed residual branches c*x; matched output at init.

    c=-1/2 or -1/(2D). The initial scalar is (1+c)^(-D), so both
    parameterizations initially output x and can represent the teacher x/4.
    This isolates conditioning; branches are fixed, without normalization.
    """
    width: int = 32
    depth: int = 1
    scaling: str = "ordinary"

    def check(self):
        require(self.width >= 32 and 1 <= self.depth <= 64, "Width >=32 and depth in [1,64]")
        require(self.scaling in ("ordinary", "depth"), "Transport scaling is ordinary or depth")

    @property
    def context(self):
        return 1

    @property
    def branch_coefficient(self):
        return -.5 / (self.depth if self.scaling == "depth" else 1)


@dataclass(frozen=True)
class GradientTargets(Benchmark):
    """Disjoint sign-vector regression; final test is sealed during training.

    Teacher branch coefficients cycle through 1/4, 1/2, 3/4. The linear
    transport teacher is x/4. No checkpoint selection or test-based stopping.
    """
    train: int = 512
    validation: int = 256
    test: int = 256

    def check(self):
        require(min(self.train, self.validation, self.test) > 0, "Every split is nonempty")

    @property
    def context(self):
        return 1

    @property
    def observed(self):
        return ("train", "validation")
