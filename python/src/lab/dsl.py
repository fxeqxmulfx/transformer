"""The experiment language: every word an experiment file uses.

    from lab.dsl import *

    experiments = {"softmax": base, "sparsemax": substitute(base, Softmax, Sparsemax())}
"""

from .domain.basis import basis
from .domain.benchmarks import ModularDivision, TinyShakespeare
from .domain.experiment import Experiment, grid
from .domain.generative import (Addition, BooleanAnd, Copy, Count, DoubleHistogram, Histogram, Mode, MostFrequent,
                                Parity, RandomLM, Reverse, Sort)
from .domain.model import (FFN, GELU, XSA, Attention, Block, FusedQKV, LayerNorm, NoPositions, Normal,
                           PerHeadQKV, PostNorm, PreNorm, QKNorm, ReLU, ReLU2, RMSNorm, RoPE, ScaledDot,
                           ScaledResidual, Sinusoidal, Softmax, Sparsemax, Tied, TorchDefault, Transformer, Untied)
from .domain.optimizers import (EVD, SGD, AdaFisher, AdaGrad, Adam, AdamNC, AdamW, AdamX, AMSGradMD, AMSGradW,
                                Chebyshev, Clipped, Constant, CoupledNewton, Dash, Geometric, Guarded, Inverse,
                                InverseSqrt, Magma, Muon, NewtonDB, RMSProp)
from .domain.spec import describe, fingerprint, substitute, swap, walk
from .domain.memorization import Memorization
from .domain.stopping import EarlyStopping, Solved
from .domain.synthetic import Synthetic
from .domain.tasks import CRASP, MQAR, AlternatingBlocks, Dyck, Lookup, TypedDyck
from .domain.training import (AttentionDiagnostics, Budget, Checkpoint, Compiled, Cosine, CudaGraph, Diagnostics, Eager, Evaluate, Schedule,
                               Seeds)

__all__ = [
    # model
    "Transformer", "Block", "Attention", "FFN", "XSA",
    "RMSNorm", "LayerNorm", "PreNorm", "PostNorm",
    "RoPE", "Sinusoidal", "NoPositions",
    "FusedQKV", "PerHeadQKV", "ScaledDot", "QKNorm", "Softmax", "Sparsemax",
    "ReLU", "ReLU2", "GELU", "Tied", "Untied", "TorchDefault", "Normal", "ScaledResidual",
    # benchmarks
    "ModularDivision", "TinyShakespeare", "Synthetic", "Memorization",
    # synthetic tasks
    "MQAR", "Lookup", "Dyck", "AlternatingBlocks", "TypedDyck", "CRASP",
    "Histogram", "DoubleHistogram", "Mode", "MostFrequent", "Copy", "Reverse", "Sort", "Count", "Addition",
    "Parity", "BooleanAnd", "RandomLM",
    # optimizers
    "SGD", "AdamW", "AMSGradW", "Adam", "AdamX", "AdaGrad", "AdamNC", "RMSProp", "Muon", "Guarded", "Magma", "Clipped",
    "Dash", "NewtonDB", "CoupledNewton", "EVD", "Chebyshev", "AdaFisher", "AMSGradMD",
    "Constant", "Inverse", "InverseSqrt", "Geometric",
    # training
    "Schedule", "Cosine", "Budget", "Seeds", "Evaluate", "Diagnostics", "AttentionDiagnostics", "Checkpoint",
    "EarlyStopping", "Solved", "Eager", "CudaGraph", "Compiled",
    # composition
    "Experiment", "grid", "swap", "substitute", "walk", "describe", "fingerprint",
    # suites
    "basis",
]
