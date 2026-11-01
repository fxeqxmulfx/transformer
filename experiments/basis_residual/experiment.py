"""Task160: original Basis problems, current GPT against conditional gains."""

from lab.dsl import *

base = {f"{size}-{task}": run
        for size, mode, model in (("small", "easy", CanonicalResidual(64, 2, 4, 129)),
                                  ("large", "hard", CanonicalResidual(128, 6, 4, 129)))
        for task, run in residual_basis(model, mode).items()}
variants = {f"{variant}-{label}": run if variant == "canonical" else substitute(
                run, CanonicalResidual, AdaptiveResidual(**vars(run.model)))
            for label, run in base.items() for variant in ("canonical", "adaptive")}
experiments = grid(variants, {"seeds.model": {"s0": 0, "s1": 1}})
