"""Optimizer blocks: the update rule an experiment trains under.

The adaptive rules follow their sources without bias correction, as
arXiv:1904.09237 and arXiv:1904.03590v4 state them; epsilon is a numerical
extension the sources do not have. Where a source lets a coefficient change
with the update t = 1, 2, ..., a `Decay` block multiplies it: `beta1_decay`
makes b1_t = b1 c_t, and `lr_decay` makes the step size alpha_t = lr c_t.
"""

from dataclasses import dataclass
import math

from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class Decay(Spec, kind=True):
    """A multiplier c_t of update t = 1, 2, ..., with c_1 = 1."""


@dataclass(frozen=True)
class Constant(Decay):
    """c_t = 1."""


@dataclass(frozen=True)
class Inverse(Decay):
    """c_t = 1 / t."""


@dataclass(frozen=True)
class InverseSqrt(Decay):
    """c_t = 1 / sqrt(t)."""


@dataclass(frozen=True)
class Geometric(Decay):
    """c_t = ratio^(t - 1)."""
    ratio: float = 0.99

    def check(self):
        require(0 < self.ratio < 1, "A geometric ratio lies strictly between zero and one")


def check_rate(lr):
    require(math.isfinite(lr) and lr > 0, "Learning rate must be positive")


def check_decays(spec, *names):
    for name in names:
        require_kind(getattr(spec, name), Decay, name)


@dataclass(frozen=True)
class Optimizer(Spec, kind=True):
    """The update rule."""


@dataclass(frozen=True)
class AdamW(Optimizer):
    """torch.optim.AdamW: bias-corrected moments and decoupled decay.

    `decay` selects the decayed tensors: "all" trainable parameters, or only
    the "matrices" (ndim >= 2) with vectors in an undecayed group.
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8
    decay: str = "all"

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas),
                "AdamW needs two betas in [0, 1)")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
        require(self.decay in ("all", "matrices"), "AdamW decay scope is 'all' or 'matrices'")


@dataclass(frozen=True)
class SGD(Optimizer):
    """Gradient descent with decoupled decay: x <- x - lr (g + weight_decay x)."""
    lr: float
    weight_decay: float = 0.0

    def check(self):
        check_rate(self.lr)
        require(self.weight_decay >= 0, "Decay must be nonnegative")


@dataclass(frozen=True)
class AMSGradW(Optimizer):
    """AMSGrad on raw moments, with decoupled decay.

    Source: arXiv:1904.03590v4, Algorithm 1 and Section 6, with the decay of
    arXiv:2606.25971v2, Sections 2 and 4.1; `Transformer.AMSGradW.trainingStep`.
    m <- b1_t m + (1 - b1_t) g, v <- b2 v + (1 - b2) g^2, vmax <- max(vmax, v),
    and x <- x - lr (c_t m / (sqrt(vmax) + eps) + weight_decay x), with c_t the
    `lr_decay`. Weight decay 0 is AMSGrad, `Transformer.AMSGrad.amsgradRule`;
    its Theorem 4.1 takes alpha_t = lr / sqrt(t) (`InverseSqrt()`) with
    b1_t = b1 / t (`Inverse()`) or b1 lambda^(t-1) (`Geometric(lambda)`).
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8
    beta1_decay: Decay = Constant()
    lr_decay: Decay = Constant()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] <= 1,
                "AMSGradW needs beta1 in [0, 1) and beta2 in [0, 1]")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class Adam(Optimizer):
    """Adam without debiasing: AMSGrad's moments, normalized by v itself.

    Source: arXiv:1904.09237, Section 2, (Adam); `Transformer.AdamBeyond.adamRule`.
    x <- x - lr c_t m / (sqrt(v) + eps), with m and v as in `AMSGradW`.
    """
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    beta1_decay: Decay = Constant()
    lr_decay: Decay = Constant()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas), "Adam needs two betas in [0, 1)")
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class AdamX(Optimizer):
    """AMSGrad whose running maximum follows the change of b1_t.

    Source: arXiv:1904.03590v4, Section 5, Algorithm 2; `Transformer.AMSGrad.adamXRule`:
    vhat_t = max(((1 - b1_t) / (1 - b1_{t-1}))^2 vhat_{t-1}, v_t), vhat_1 = v_1,
    and x <- x - lr c_t m / (sqrt(vhat) + eps). Its Theorem 5.1 takes
    alpha_t = lr / sqrt(t) with any b1_t <= b1, and its Corollary 5.6
    b1_t = b1 lambda^(t-1) or b1 / t, the default.
    """
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    beta1_decay: Decay = Inverse()
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas), "AdamX needs two betas in [0, 1)")
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class AdaGrad(Optimizer):
    """AdaGrad: the mean of the squared gradients at step size lr / sqrt(t).

    Source: arXiv:1904.09237, Section 2, (AdaGrad); `Transformer.AdamBeyond.adagradRule`:
    vhat <- ((t - 1) vhat + g^2) / t and x <- x - lr c_t g / (sqrt(vhat) + eps),
    so that the default c_t = 1 / sqrt(t) moves by lr g / sqrt(sum of g^2).
    """
    lr: float
    eps: float = 1e-8
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "lr_decay")


@dataclass(frozen=True)
class AdamNC(Optimizer):
    """AdamNC with b2_t = 1 - 1/t: momentum over AdaGrad's mean of squared gradients.

    Source: arXiv:1904.09237, appendix, Algorithm 3, `Transformer.AdamBeyond.adamNCRule`:
    m <- b1_t m + (1 - b1_t) g, vhat as in `AdaGrad`, x <- x - lr c_t m / (sqrt(vhat) + eps).
    Its Theorem 5 takes alpha_t = lr / sqrt(t), and the remark after its
    Corollary 2 the decay b1_t = b1 / t; both are the defaults.
    """
    lr: float
    beta1: float = 0.9
    eps: float = 1e-8
    beta1_decay: Decay = Inverse()
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(0 <= self.beta1 < 1 and self.eps > 0, "AdamNC needs beta1 in [0, 1) and a positive epsilon")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class RMSProp(Optimizer):
    """RMSProp on the raw second moment, without momentum or bias correction.

    v <- b2 v + (1 - b2) g^2 and x <- x - lr g / (sqrt(v) + eps): the dense
    baseline of arXiv:2602.15322v1, Section 2, as its benchmark ran it. The
    manuscript is not among papers/; the rule is the benchmark's.
    """
    lr: float
    beta2: float = 0.999
    eps: float = 1e-8

    def check(self):
        check_rate(self.lr)
        require(0 <= self.beta2 < 1 and self.eps > 0, "RMSProp needs beta2 in [0, 1) and a positive epsilon")


@dataclass(frozen=True)
class Muon(Optimizer):
    """Muon on the hidden matrices, bias-corrected Adam on every other parameter.

    Source: arXiv:2502.16982, Sections 2.1-2.2; `Transformer.Muon.muonStep`
    without decay. A matrix of the blocks moves by 0.2 sqrt(max(a, b)) times
    five Newton-Schulz steps (3.4445, -4.775, 2.0315) on the Frobenius-
    normalized Nesterov input mu M + g, after M <- mu M + g. The embedding, a
    readout and the vectors move by Adam with bias correction (0.9, 0.999,
    1e-8) at `auxiliary` times the rate, as the optimizer benchmark ran them.
    """
    lr: float
    momentum: float = 0.95
    auxiliary: float = 0.05

    def check(self):
        check_rate(self.lr)
        require(0 <= self.momentum < 1 and self.auxiliary > 0,
                "Muon needs a momentum in [0, 1) and a positive auxiliary rate")


@dataclass(frozen=True)
class Guarded(Optimizer):
    """The descent guard over another rule, on the joint parameter vector.

    Source: `Transformer.Optimization.descentGuard`: the rule's direction d is
    kept when <g, d> >= sigma |g|^2 and |d| <= |g|, and the gradient g is used
    otherwise. The rule's state advances either way, so a rule whose every
    direction is rejected trains as SGD
    (`Transformer.OptimizerBenchmark.guardedBatchRun_eq_sgd`). Over
    `AMSGradMD`, which writes new values, d is their displacement over lr.
    """
    base: Optimizer
    sigma: float = 0.5

    def check(self):
        require_kind(self.base, Optimizer, "base")
        require(not any(isinstance(base, Guarded) for base in bases(self)), "The guard applies once")
        require(0 < self.sigma <= 1, "The guard's sigma lies in (0, 1]")

    @property
    def lr(self):
        return self.base.lr


@dataclass(frozen=True)
class Magma(Optimizer):
    """MAGMA over a direction rule: hidden matrices masked at random, damped by alignment.

    Source: arXiv:2602.15322v1, Sections 2-3 and Algorithm 1;
    `Transformer.Magma.cosine`, `damping` and `algorithmDisplacement`. After
    the base computes every direction, each hidden matrix W, a weight of the
    blocks, updates its scale s <- 0.9 s + 0.1 sigmoid(cos(m, g) / tau) from
    0.5, where m is the base's raw first moment of W when it keeps one (Muon's
    momentum, the m of the Adam family and of AdaFisher) and otherwise an EMA
    m <- 0.9 m + 0.1 g that MAGMA keeps, and cos is zero at a zero vector. W
    then moves by s times its direction with probability `survival`, and stays
    otherwise: without the 1 / survival of Section 5's analysis. The masks
    come from a CPU generator seeded by `seed`, or by the model seed + 20000
    when it is None, the benchmark's convention. The manuscript is not among
    papers/.
    """
    base: Optimizer
    tau: float = 2.0
    survival: float = 0.5
    seed: int | None = None

    def check(self):
        require_kind(self.base, Optimizer, "base")
        require(not any(isinstance(base, Magma) for base in bases(self)), "MAGMA applies once")
        require(not any(isinstance(base, AMSGradMD) for base in bases(self)),
                "MAGMA damps the directions of a rule, and AMSGradMD writes new values")
        require(math.isfinite(self.tau) and self.tau > 0, "MAGMA's temperature is finite and positive")
        require(0 < self.survival <= 1, "MAGMA's survival probability lies in (0, 1]")
        require(self.seed is None or self.seed >= 0, "Seeds are nonnegative")

    @property
    def lr(self):
        return self.base.lr


@dataclass(frozen=True)
class Clipped(Optimizer):
    """Another rule, on the gradient rescaled to a global norm of at most `norm`.

    Source: norm clipping, arXiv:1211.5063, Section 3.2 and Algorithm 1 (not
    in `papers/`), as torch's `clip_grad_norm_` computes it and the
    historical synthetic trainer applied it (`grad_clip`): the gradient g of
    all trainable parameters, as one vector, is multiplied by
    min(1, norm / (|g| + 1e-6)), where the paper rescales by norm / |g| only
    when |g| >= norm. It acts before the rule and every stage of it, so it is
    written outermost. The recorded gradient norm is |g| before clipping.
    """
    base: Optimizer
    norm: float = 1.0

    def check(self):
        require_kind(self.base, Optimizer, "base")
        require(math.isfinite(self.norm) and self.norm > 0, "The clipping norm is finite and positive")

    @property
    def lr(self):
        return self.base.lr


def clipping(optimizer):
    """The global gradient norm an experiment's `optimizer` clips to: infinite unless it is `Clipped`."""
    return optimizer.norm if isinstance(optimizer, Clipped) else math.inf


def bases(stage):
    """The optimizers a stage is built on, from its own base inward."""
    while isinstance(stage, (Guarded, Magma)):
        stage = stage.base
        yield stage


@dataclass(frozen=True)
class InverseRoot(Spec, kind=True):
    """A solver for (A + eps I)^(-1/4) of every Shampoo history A of a batch."""


@dataclass(frozen=True)
class EVD(InverseRoot):
    """The root from an eigendecomposition; torch cannot capture it in a CUDA graph."""


@dataclass(frozen=True)
class NewtonDB(InverseRoot):
    """Two chained Newton-Denman-Beavers square roots of `steps` iterations each.

    `Transformer.DASH.countedBatchPiInverseFourth`, on the matrix scaled by a
    guarded estimate from three power iterations of the ones vector.
    """
    steps: int = 6

    def check(self):
        require(self.steps >= 1, "A Newton solver takes at least one step")


@dataclass(frozen=True)
class CoupledNewton(InverseRoot):
    """`steps` coupled Newton iterations for the inverse fourth root.

    `Transformer.DASH.countedBatchPiCnFour`, on the matrix scaled as for `NewtonDB`.
    """
    steps: int = 8

    def check(self):
        require(self.steps >= 1, "A Newton solver takes at least one step")


@dataclass(frozen=True)
class Chebyshev(InverseRoot):
    """The cosine fit of degree `degree` on `samples` nodes, evaluated by Clenshaw.

    `Transformer.DASH.countedBatchPiPower`, on the matrix scaled as for
    `NewtonDB`; fewer samples than degree + 1 are raised to degree + 1.
    """
    degree: int = 60
    samples: int = 1000

    def check(self):
        require(self.degree >= 0 and self.samples >= 1, "A cosine fit needs a degree and samples")


@dataclass(frozen=True)
class Dash(Optimizer):
    """Blocked Shampoo with Adam grafting, batched over the blocks of one shape.

    Source: arXiv:2602.02016v2, Sections 2-4; `Transformer.DASH.leftEma`,
    `rightEma`, `preconditionedGradient` and `graft`. Every parameter is cut
    into blocks of at most `block` x `block`, a vector as one column. Per block
    G, L <- b L + (1 - b) G G^T and R <- b R + (1 - b) G^T G, and the
    direction is (L + eps I)^(-1/4) G (R + eps I)^(-1/4) rescaled to the
    Frobenius norm of Adam's m / (sqrt(v) + eps) under `graft_betas`, and zero
    where the Shampoo direction is zero.
    """
    lr: float
    solver: InverseRoot = NewtonDB()
    block: int = 32
    beta: float = 0.99
    graft_betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-4

    def check(self):
        check_rate(self.lr)
        require_kind(self.solver, InverseRoot, "solver")
        require(self.block >= 1 and 0 <= self.beta < 1 and self.eps > 0,
                "Dash needs a positive block size, beta in [0, 1) and a positive epsilon")
        require(len(self.graft_betas) == 2 and all(0 <= beta < 1 for beta in self.graft_betas),
                "Dash needs two graft betas in [0, 1)")


@dataclass(frozen=True)
class AdaFisher(Optimizer):
    """Momentum preconditioned by a damped Kronecker-factored Fisher diagonal.

    Source: arXiv:2405.16397v3, Sections 3.2-3.3 and Algorithm 1, as
    corrected in `Transformer.AdaFisher.factorEMA`, `minMaxDiagonal`,
    `fisherDiagonal`, `correctedMomentum` and `parameterStep`. For a linear
    weight, h and s are the squared inputs and squared output gradients of the
    batch summed over positions, H <- gamma h + (1 - gamma) H and likewise S,
    f = minmax(S) minmax(H)^T + damping, and after m <- beta m + (1 - beta) g
    the weight moves by lr (m / (1 - beta^t) / f + weight_decay x): AdaFisherW
    when `weight_decay` is positive. A weight shared with a linear layer, as a
    tied embedding is with the readout, takes that layer's factors. Every other
    parameter, a temperature or an untied embedding, takes ones, as the
    benchmark did: they normalize to zero, and f is the damping. The manuscript
    is not among papers/.
    """
    lr: float
    beta: float = 0.9
    gamma: float = 0.8
    damping: float = 1e-3
    weight_decay: float = 0.0

    def check(self):
        check_rate(self.lr)
        require(0 <= self.beta < 1 and 0 < self.gamma <= 1 and self.damping > 0 and self.weight_decay >= 0,
                "AdaFisher needs beta in [0, 1), gamma in (0, 1], a positive damping and a nonnegative decay")


@dataclass(frozen=True)
class AMSGradMD(Optimizer):
    """AMSGrad on the magnitude-direction factorization of the hidden matrices.

    Source: arXiv:2606.25971v2, Section 3.1 and Appendix A, Algorithm 2, with
    AMSGrad as arXiv:1904.03590v4, Algorithm 1 states it;
    `Transformer.MagnitudeDirection.amsgradMDProposal`. A hidden matrix W, a
    weight of the blocks, is stored as softplus(r)_i D_ij softplus(c)_j, with
    raw gains starting at softplus^-1(1) and D on the sphere of radius |W_0|.
    An update moves D at `lr`, and r and c at `gain_rate`, each by an AMSGrad
    of its own on the gradients at the old factors, projects D back onto the
    sphere (`matrixProject`, zero at zero) and writes W fused from the three;
    every other parameter moves by AMSGrad at `auxiliary_rate`.

    Under `Guarded`, the benchmark's extension `checkedProposal`, which the
    manuscript does not have, D may move at a `direction_rate` of its own: the
    proposal's displacement over `lr` is the direction the guard checks, with
    every value finite and no matrix zero, and a rejected proposal becomes a
    gradient step at `lr`, halved for a matrix it would zero, with the row
    gains rebalanced onto the sphere (`rebalanceStorage`). The manuscript is
    not among papers/.
    """
    lr: float
    direction_rate: float | None = None
    gain_rate: float = 1e-3
    auxiliary_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def check(self):
        for rate in (self.lr, self.gain_rate, self.auxiliary_rate):
            check_rate(rate)
        if self.direction_rate is not None:
            check_rate(self.direction_rate)
        require(len(self.betas) == 2 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] <= 1,
                "AMSGradMD needs beta1 in [0, 1) and beta2 in [0, 1]")
        require(self.eps > 0, "Epsilon must be positive")
