"""PyTorch optimizers with explicit Lean recurrence correspondence.

Importing this package needs only PyTorch and Python's standard library.
See README.md for usage and LEAN_CORRESPONDENCE.md for certification limits.
"""

from __future__ import annotations

from torch import optim as standard

from .api import (
    SGD,
    AdaGrad,
    Adam,
    AdamNC,
    AdamW,
    AdamWDirection,
    AdamX,
    AMSGrad,
    AMSGradW,
    Clipped,
    RMSProp,
)
from .atomic import AtomicColumns
from .config import (
    EVD,
    Chebyshev,
    Constant,
    CoupledNewton,
    Geometric,
    Inverse,
    InverseSqrt,
    NewtonDB,
)
from .geometry import AdaFisher, AMSGradMD, Dash, Guarded, Magma, Muon
from .mdw import AdamMDW, AMSGradMDW
from .optimistic import ClippedOptimistic
from .schedule_free import (
    ScheduleFreeAdamW,
    ScheduleFreeAdamWPaper,
    ScheduleFreeAMSGradW,
    ScheduleFreeSGD,
)
from .schedule_free_md import ScheduleFreeAMSGradMD
from .schedule_free_mdw import ScheduleFreeAdamMDW, ScheduleFreeAMSGradMDW
from .schedule_free_muon import ScheduleFreeMuon
from .schedules import LearningRateSchedule, LinearWarmup

# Explicit names for paper conventions that differ from torch.optim defaults.
PaperAdam = Adam
PaperAMSGrad = AMSGrad
PaperAdaGrad = AdaGrad

__version__ = "0.1.0"
__all__ = [
    "EVD",
    "SGD",
    "AMSGrad",
    "AMSGradMD",
    "AMSGradMDW",
    "AMSGradW",
    "AdaFisher",
    "AdaGrad",
    "Adam",
    "AdamMDW",
    "AdamNC",
    "AdamW",
    "AdamWDirection",
    "AdamX",
    "AtomicColumns",
    "Chebyshev",
    "Clipped",
    "ClippedOptimistic",
    "Constant",
    "CoupledNewton",
    "Dash",
    "Geometric",
    "Guarded",
    "Inverse",
    "InverseSqrt",
    "LearningRateSchedule",
    "LinearWarmup",
    "Magma",
    "Muon",
    "NewtonDB",
    "PaperAMSGrad",
    "PaperAdaGrad",
    "PaperAdam",
    "RMSProp",
    "ScheduleFreeAMSGradMD",
    "ScheduleFreeAMSGradMDW",
    "ScheduleFreeAMSGradW",
    "ScheduleFreeAdamMDW",
    "ScheduleFreeAdamW",
    "ScheduleFreeAdamWPaper",
    "ScheduleFreeMuon",
    "ScheduleFreeSGD",
    "standard",
]
