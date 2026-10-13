"""The verified shared embedding/attention replacement, with deferred FFN."""

from dataclasses import dataclass
import math

from .model import Model
from .spec import require
from .generative import Parity
from .synthetic import Synthetic
from .tasks import AlternatingBlocks, MQAR


@dataclass(frozen=True)
class TensorStack(Model):
    """Actual causal tensor stack from Lean's Structured.TensorBasisModel.

    Source: tensorFullStack, tensorFunction, basisTensor_solves and
    basisStackBatchNLL_convex at b0a43a8. Width is the ordinary residual
    stream width; two shared structured heads replace softmax attention.
    All 52 token fields, initial/transition/value/branch logits and learned
    absolute/relative positions are free. Ten tied decoder axes and a unit
    anchor are architectural constants. Both original ReLU2 FFN matrices
    are fixed zero. The objective is the complete data-generated Basis
    likelihood, not output cross entropy. No semantic rule enters forward.

    Deviations: floating-point arithmetic, stable log-sum-exp contraction,
    Gaussian initialization instead of the finite capability witness.
    The implementation explicitly executes every residual layer and zero
    FFN; no layer or preprocessing cost is assigned to imaginary work.

    """
    width: int
    depth: int
    context: int
    eps: float = 1e-5
    std: float = 0.02

    def check(self):
        require(type(self.width) is int and self.width >= 64,
                "TensorStack needs at least 64 residual coordinates")
        require(type(self.depth) is int and self.depth >= 1, "Depth must be positive")
        require(type(self.context) is int and self.context >= 1, "Context must be positive")
        require(math.isfinite(self.eps) and self.eps > 0, "RMS epsilon must be positive and finite")
        require(math.isfinite(self.std) and self.std > 0, "Initialization scale must be positive and finite")

    def parameter_count(self, vocab):
        """BindingParameters' exact dimension; independent of dataset size."""
        require(type(vocab) is int and 1 <= vocab <= 1024, "The verified decoder covers 1..1024 tokens")
        return 52 * vocab + 3 * self.context + 128


@dataclass(frozen=True)
class TensorGain(TensorStack):
    """The actual verified tensor stack in fixed linear potential coordinates.

    Source: TensorGain.tensorGainFunction, basisGain_solves and
    basisGainBatchNLL_convex at c12df24. Every free potential is gain *
    its optimizer coordinate. This invertible linear parameterization
    changes conditioning and keeps the model class, parameter count,
    ordinary AdamW and true inference/loss coupling. Initialize coordinates
    at std/gain to keep the actual-potential initialization scale std.
    Deviations: float32, stable contractions and Gaussian initialization
    inherited from TensorStack. Successful learning, floating-point
    identities and AdamW convergence remain experimental.
    """
    gain: float = 8.0

    def check(self):
        super().check()
        require(math.isfinite(self.gain) and self.gain > 0, "Fixed potential gain must be positive and finite")


def check_tensor_benchmark(benchmark):
    """Reject supervision outside the independently proved Basis grammars."""
    require(isinstance(benchmark, Synthetic) and benchmark.study is None,
            "TensorStack complete supervision needs a clean synthetic Basis task")
    task = benchmark.task
    if isinstance(task, AlternatingBlocks):
        require(task.blocks in (2, 4) and benchmark.context <= 128,
                "Verified depth has two/four alternating runs and context at most 128")
    elif isinstance(task, MQAR):
        require(task.symbols == 256 and (task.pairs, task.overwrites) in ((8, 0), (16, 8))
                and task.queries == 8 and benchmark.context <= 64,
                "Verified recall uses the actual eight/sixteen-write Basis grammars")
    elif isinstance(task, Parity):
        require(task.scratchpad == "none" and not task.hints and task.symbols == 32
                and benchmark.context <= 19, "Verified parity has 1..16 bits and no scratchpad/hints")
    else:
        raise ValueError("No verified complete labels for this task")
