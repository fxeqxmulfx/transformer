"""Softmax against sparsemax on the calibrated basis; EXPERIMENT_PLAN.md, steps 0 and 1."""

from lab.dsl import *


def gptmini(width, depth):
    """The four-head GPTMini of experiments/basis at bd63e50, with the same blocks."""
    attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=width, depth=depth, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(0.02), context=64)


MODELS = {"small": gptmini(64, 2), "large": gptmini(128, 6)}
RATES = {"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3, "1e-2": 1e-2}
THREADS = {("large", "recall"): 4, ("large", "depth"): 2, ("large", "parity"): 2}
WEIGHTS = {"softmax": Softmax(), "sparsemax": Sparsemax()}


def measured(run, name, task):
    """Record the last training batch's position losses and head scales at every observation."""
    run = swap(run, "execution.threads", THREADS.get((name, task), 1))
    return swap(run, "diagnostics", Diagnostics(every=run.evaluate.every))


runs = {f"{weights}-{mode}-{name}-{task}-seed{seed}": measured(run, name, task)
        for weights, normalizer in WEIGHTS.items() for mode in ("easy", "hard")
        for name, model in MODELS.items() for seed in (0, 1, 2)
        for task, run in basis(substitute(model, Softmax, normalizer), mode, seed).items()
        if (mode, task) != ("hard", "parity")}


def timed(run):
    """300 compiled updates at the task's batch and rate, observed and sampled every 100."""
    run = swap(swap(run, "budget.updates", 300), "evaluate.every", 100)
    return swap(swap(run, "diagnostics", Diagnostics(every=100)), "stopping", None)


times = {f"time-{weights}-small-{task}": timed(runs[f"{weights}-easy-small-{task}-seed0"])
         for weights in WEIGHTS for task in ("depth", "recall", "parity")}

experiments = {**runs, **times}
