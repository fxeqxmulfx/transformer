"""PyTorch optimizers with explicit Lean recurrence correspondence.

Importing this package needs only PyTorch and Python's standard library.
See README.md for usage and LEAN_CORRESPONDENCE.md for certification limits.
"""

from torch import optim as standard

from .api import AdaGrad, Adam, AdamNC, AdamW, AdamWDirection, AdamX, AMSGrad, AMSGradW, Clipped, RMSProp, SGD
from .atomic import AtomicColumns
from .config import Chebyshev, Constant, CoupledNewton, EVD, Geometric, Inverse, InverseSqrt, NewtonDB
from .geometry import AdaFisher, AMSGradMD, Dash, Guarded, Magma, Muon
from .optimistic import ClippedOptimistic
from .schedule_free import ScheduleFreeAdamW, ScheduleFreeSGD

# Explicit names for paper conventions that differ from torch.optim defaults.
PaperAdam = Adam
PaperAMSGrad = AMSGrad
PaperAdaGrad = AdaGrad

__version__ = "0.1.0"
__all__ = ["AdaGrad", "AdaFisher", "Adam", "AdamNC", "AdamW", "AdamWDirection", "AdamX", "AMSGrad", "AMSGradW",
           "AMSGradMD", "AtomicColumns", "Clipped", "ClippedOptimistic", "Dash", "Guarded", "Magma", "Muon",
           "RMSProp", "SGD", "ScheduleFreeAdamW", "ScheduleFreeSGD", "Chebyshev", "Constant", "CoupledNewton", "EVD",
           "Geometric", "Inverse", "InverseSqrt", "NewtonDB", "PaperAdam", "PaperAMSGrad", "PaperAdaGrad", "standard"]
