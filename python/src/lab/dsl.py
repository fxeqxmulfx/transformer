"""The experiment language: every word an experiment file uses.

    from lab.dsl import *

    experiments = {"softmax": base, "sparsemax": substitute(base, Softmax, Sparsemax())}
"""

from .domain.basis import basis
from .domain.atomic import (AtomicColumns, AtomicMatching, BindingPricing, MatchingBindings,
                           MatchingOrders, OrderPricing, PairedMatching, SearchPricing)
from .domain.benchmarks import ModularDivision, TinyShakespeare
from .domain.context import extend_context
from .domain.grokking import GrokkingDiagnostics
from .domain.experiment import Experiment, grid
from .domain.generative import (Addition, BooleanAnd, Copy, Count, DoubleHistogram, Histogram, Mode, MostFrequent,
                                Parity, RandomLM, Reverse, Sort)
from .domain.model import (FFN, GELU, XSA, Attention, Block, FusedQKV, LayerNorm, NoPositions, Normal,
                           LearnedScaledDot, PerHeadQKV, PostNorm, PreNorm, QKNorm, ReLU, ReLU2, RMSNorm, RoPE, ScaledDot,
                           ScaledResidual, SiLU, Sinusoidal, Softmax, Sparsemax, SurrogateWeights, Tied, TorchDefault, Transformer, Untied, YaRN)
from .domain.rotary import DenseBranch, LabDivision, Rotate90, SwiGLU
from .domain.paired import CrossEntropyAnswers, DivisionPairs, SquaredAnswers, VectorReadout
from .domain.regression import MLP, ModularRegression, OneHot, Scalar
from .domain.optimizers import (ANSR, EVD, SGD, AdaFisher, AdaGrad, Adam, AdamNC, AdamW, AdamX, AMSGradMD, AMSGradW,
                                Chebyshev, Clipped, Constant, CoupledNewton, Dash, Geometric, Guarded, Inverse,
                                InverseSqrt, Magma, Muon, NewtonDB, RMSProp)
from .domain.mdw import AMSGradMDW, AdamMDW, ScheduleFreeAMSGradMDW, ScheduleFreeAdamMDW
from .domain.schedule_free import ScheduleFreeAdamW, ScheduleFreeMuon
from .domain.spec import describe, fingerprint, substitute, swap, walk
from .domain.memorization import Memorization
from .domain.stopping import EarlyStopping, Solved
from .domain.synthetic import Synthetic
from .domain.tensor import TensorGain, TensorStack
from .domain.tasks import CRASP, MQAR, AlternatingBlocks, Dyck, Lookup, TypedDyck
from .domain.training import (AttentionDiagnostics, Budget, FlopBudget, Checkpoint, Compiled, Measured, Cosine, CudaGraph, Diagnostics, Eager, Evaluate, Schedule,
                               Seeds)

__all__ = [
    # model
    "Transformer", "TensorStack", "TensorGain", "AtomicMatching", "PairedMatching", "MLP", "Scalar", "OneHot", "Block", "Attention", "FFN", "XSA",
    "RMSNorm", "LayerNorm", "PreNorm", "PostNorm",
    "RoPE", "YaRN", "Sinusoidal", "NoPositions",
    "FusedQKV", "PerHeadQKV", "ScaledDot", "LearnedScaledDot", "QKNorm", "Softmax", "Sparsemax", "SurrogateWeights",
    "ReLU", "ReLU2", "GELU", "SiLU", "Tied", "Untied", "TorchDefault", "Normal", "ScaledResidual",
    "Rotate90", "DenseBranch", "SwiGLU",
    "VectorReadout", "SquaredAnswers", "CrossEntropyAnswers",
    # benchmarks
    "ModularDivision", "ModularRegression", "LabDivision", "TinyShakespeare", "Synthetic", "Memorization", "MatchingOrders", "MatchingBindings",
    "DivisionPairs",
    # synthetic tasks
    "MQAR", "Lookup", "Dyck", "AlternatingBlocks", "TypedDyck", "CRASP",
    "Histogram", "DoubleHistogram", "Mode", "MostFrequent", "Copy", "Reverse", "Sort", "Count", "Addition",
    "Parity", "BooleanAnd", "RandomLM",
    # optimizers
    "ANSR", "AtomicColumns", "OrderPricing", "BindingPricing", "SearchPricing", "SGD", "AdamW", "AMSGradW", "Adam", "AdamX", "AdaGrad", "AdamNC", "RMSProp", "Muon", "Guarded", "Magma", "Clipped",
    "Dash", "NewtonDB", "CoupledNewton", "EVD", "Chebyshev", "AdaFisher", "AMSGradMD",
    "AMSGradMDW", "AdamMDW", "ScheduleFreeAdamW", "ScheduleFreeMuon", "ScheduleFreeAMSGradMDW", "ScheduleFreeAdamMDW",
    "Constant", "Inverse", "InverseSqrt", "Geometric",
    # training
    "Schedule", "Cosine", "Budget", "FlopBudget", "Seeds", "Evaluate", "Diagnostics", "AttentionDiagnostics", "GrokkingDiagnostics", "Checkpoint",
    "EarlyStopping", "Solved", "Eager", "CudaGraph", "Compiled", "Measured",
    # composition
    "Experiment", "grid", "swap", "substitute", "walk", "describe", "fingerprint", "extend_context",
    # suites
    "basis",
]
