"""Task158: matched current GPT and conditional post-RMS gains, first99% stop."""

from lab.dsl import *

base = Experiment(
    model=CanonicalResidual(width=128, depth=4, heads=4, context=33),
    benchmark=ResidualPairs(prime=97, operation="division", target_accuracy=.99),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1, warmup=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=100000, batch=256),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=1000, batch=512),
    checkpoint=Checkpoint(every=1000),
    diagnostics=Diagnostics(gradients=True),
    execution=ResidualMaxAutotune(device="cuda", attention_backend="math"),
)

variants = {"canonical": base, "adaptive": substitute(
    base, CanonicalResidual, AdaptiveResidual(width=128, depth=4, heads=4, context=33))}
experiments = grid(variants, {"seeds.model": {"s11": 11, "s17": 17}})
