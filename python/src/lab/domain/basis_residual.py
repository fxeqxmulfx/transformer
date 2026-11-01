"""Task160 recipes: retain Basis tasks and compare two current GPT parameterizations."""

from .basis import BASIS, OBSERVED, WARMUP
from .experiment import Experiment
from .residual_basis import ResidualBasis
from .residual_transformer import ResidualMaxAutotune
from .schedule_free import ScheduleFreeAdamW
from .spec import require, swap
from .training import Budget, Checkpoint, Diagnostics, Evaluate, Schedule, Seeds


def residual_basis(model, mode, seed=0):
    """Standard Basis batches/rates, long chunks and sealed first99% endpoints.

    Source: domain.basis.basis. Deviations: Schedule-Free AdamW, CUDA fullgraph
    max-autotune with math SDPA, 100000-update continuation chunks, and eval
    batch64 for the4GiB GPU. Data, training batches, warmup50 and observation
    cadence remain the original Basis ones. This does not compare the new
    optimizer's wall time causally against historical AdamW runs.
    """
    require(mode in BASIS, "Basis mode must be easy or hard")
    return {name: Experiment(
        model=swap(model, "context", benchmark.context),
        benchmark=ResidualBasis(**vars(benchmark)),
        optimizer=ScheduleFreeAdamW(lr=recipe.lr, betas=(.9, .98), weight_decay=.1),
        schedule=Schedule(warmup=WARMUP), budget=Budget(100000, recipe.batch),
        seeds=Seeds(model=seed, data=1), evaluate=Evaluate(OBSERVED // recipe.batch, 64),
        checkpoint=Checkpoint(OBSERVED // recipe.batch),
        diagnostics=Diagnostics(every=OBSERVED // recipe.batch, gradients=True),
        execution=ResidualMaxAutotune(device="cuda", attention_backend="math"),
    ) for name, (benchmark, recipe) in BASIS[mode].items()}
