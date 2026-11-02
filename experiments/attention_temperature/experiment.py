"""Task162: change only learned per-head score temperature in current GPT."""

from lab.dsl import *

base = residual_basis(CanonicalResidual(128, 6, 4, 129), "hard")["recall"]
candidate = substitute(base, CanonicalResidual, TemperatureResidual(**vars(base.model)))
experiments = grid({"temperature-hard-recall": candidate}, {"seeds.model": {"s0": 0, "s1": 1}})
