"""First-crossing Basis controls using the original declarative transformer."""

from dataclasses import dataclass

from .residual_basis import ResidualBasis


@dataclass(frozen=True)
class BasisControl(ResidualBasis):
    """Task162: original GPTMini with the task160 stopping/data contract.

    Sources: experiments/basis at bd63e509 and ResidualBasis at23154349.
    Data, sequence scoring and free generation stay unchanged. Deviations
    from historical Basis: Schedule-Free AdamW, first-crossing current state,
    no best-state restore and no final-test access before cohort closure.
    The model keeps the original optimized supervised-position readout.
    """
