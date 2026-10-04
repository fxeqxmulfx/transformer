"""Softmax against sparsemax on the calibrated basis; EXPERIMENT_PLAN.md, steps 0 to 2."""

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


def neighbors(run):
    """The two rates of the basis's RATES beside a run's recipe, for EXPERIMENT_PLAN.md, H5."""
    ordered = tuple(RATES.items())
    chosen = next(index for index, (_, rate) in enumerate(ordered) if rate == run.optimizer.lr)
    return {f"lr{label}": rate for index, (label, rate) in enumerate(ordered) if abs(index - chosen) == 1}


rates = {variant: candidate for label, run in runs.items() if label.startswith("sparsemax-")
         for variant, candidate in grid({label: run}, {"optimizer.lr": neighbors(run)}).items()}

def observed(run):
    """256 fixed selection examples at every observation; EXPERIMENT_PLAN.md, step 2."""
    diagnostic = run.diagnostics
    return swap(run, "diagnostics", AttentionDiagnostics(every=diagnostic.every,
                neighbors=diagnostic.neighbors, gradients=diagnostic.gradients, examples=256))


probes = {f"probe-{label}": observed(run) for label, run in runs.items()
          if "-hard-large-recall-" in label}


def initialized(run):
    """Retain update zero and one zero-rate warmup update, without early stopping, for H2."""
    run = swap(swap(swap(run, "budget.updates", 1), "evaluate.every", 1), "stopping", None)
    return swap(observed(run), "diagnostics.every", 1)


initial = grid({f"init-{weights}-{name}-seed{seed}": initialized(runs[f"{weights}-hard-{name}-recall-seed{seed}"])
                for weights in WEIGHTS for name in MODELS for seed in (0, 1, 2)},
               {"model.block.attention.scores": {"qknorm": QKNorm(), "scaleddot": ScaledDot()}})

experiments = {**runs, **times, **rates, **probes, **initial}
