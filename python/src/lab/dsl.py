"""The experiment language: every word an experiment file uses.

    from lab.dsl import *

    experiments = {"softmax": base, "sparsemax": substitute(base, Softmax, Sparsemax())}
"""

from .domain.benchmarks import ModularDivision
from .domain.experiment import Experiment, grid
from .domain.model import (FFN, GELU, XSA, Attention, Block, FusedQKV, LayerNorm, NoPositions, Normal,
                           PerHeadQKV, PostNorm, PreNorm, QKNorm, ReLU, ReLU2, RMSNorm, RoPE, ScaledDot,
                           Sinusoidal, Softmax, Sparsemax, Tied, TorchDefault, Transformer, Untied)
from .domain.spec import describe, fingerprint, substitute, swap, walk
from .domain.training import (SGD, AdamW, AMSGradW, Budget, Checkpoint, Cosine, CudaGraph, Diagnostics, Eager,
                              Evaluate, Schedule, Seeds)

__all__ = [
    # model
    "Transformer", "Block", "Attention", "FFN", "XSA",
    "RMSNorm", "LayerNorm", "PreNorm", "PostNorm",
    "RoPE", "Sinusoidal", "NoPositions",
    "FusedQKV", "PerHeadQKV", "ScaledDot", "QKNorm", "Softmax", "Sparsemax",
    "ReLU", "ReLU2", "GELU", "Tied", "Untied", "TorchDefault", "Normal",
    # benchmarks
    "ModularDivision",
    # training
    "AdamW", "AMSGradW", "SGD", "Schedule", "Cosine", "Budget", "Seeds", "Evaluate", "Diagnostics", "Checkpoint",
    "Eager", "CudaGraph",
    # composition
    "Experiment", "grid", "swap", "substitute", "walk", "describe", "fingerprint",
]
