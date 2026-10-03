"""GPTMini on the synthetic suite under AMSGradW: the softmax baseline and its studies.

`baseline.py` of `experiments/synthetic_trainers` and its recipes
(`baseline_recipes.py`), in the experiment language: 61 runs of GPTMini of
width 64, with 2 layers of 4 heads and the context of its benchmark, under
AMSGradW at rate 1e-3 with betas (0.9, 0.999), epsilon 1e-8 and decay 0.1,
on 32 rows per update, observed in chunks of 32, each under the
double-descent study with curve tolerance 0.02 on data seed 1. A run draws
its batch order from its model seed, as `train_run` drew it.

- suite: each task of the suite and each named control of it, 1000 updates
  on 128 training rows from model seed 0, observed every 100 updates on 32
  validation rows and tested on 64: at lengths 8 to 16, tested at 32 and 64;
  MQAR and lookup at length 24, with their 4 associations and 2 queries,
  tested at 48 and 96; Dyck and typed Dyck from length 12; addition of 2 to
  4 digits, tested at 8 and 16.
- transitions: copy, and parity with and without a running scratchpad, 5000
  updates on disjoint pools of 64 training, 64 validation and 128 test rows,
  observed every 250 and tested at 16 and 32, from model seeds 0, 1 and 2.
- capacity: parity on the same pools with 20% of its training labels
  corrupted (noise seed 2), 1000 updates observed every 100, at widths 16,
  32, 64 and 128, from model seeds 0, 1 and 2.
- control: the random control on 8 training rows, 64 validation and 128
  test, 1000 updates, tested at 16 and 32, from model seeds 0, 1 and 2.

The archived runs (RTX 3050, 2026-10-01, issued eagerly;
`experiments/synthetic_trainers/baselines/amsgradw_softmax_20261002`) fit
their training splits, but for two parity runs of the suite, one run at
width 128 under label noise, and the random controls, which fit one of their
8 answers. Few generalized in distribution. On the suite the last models
reached test sequence accuracy 1.0 on both Boolean-and controls, 0.95 on
Dyck, 0.92 on alternating blocks and 0.91 on C-RASP of depths 1 and 2; 0.70
at depth 3, 0.64 and 0.66 on typed Dyck, 0.56 on parity without a
scratchpad, near the chance of its one-label answer, and 0.39 on mode; and
at most 0.19 on every other task, MQAR and lookup included. Under the
transition budget parity with a running scratchpad generalized (0.91 to
0.99), parity without it reached 0.35 to 0.44, and copy at most 0.04. Under
label noise every width tested at 0.35 to 0.56, without a trend in width.

The runs replay from CUDA graphs here, so none is its archived run bit for
bit.
"""

from lab.dsl import *

gptmini = Transformer(
    width=64, depth=2, context=64,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

study = Memorization(tolerance=0.02)


def run(benchmark, updates=1000, every=100):
    """GPTMini on `benchmark`, through a context of the benchmark's own size."""
    return Experiment(
        model=swap(gptmini, "context", benchmark.context),
        benchmark=benchmark,
        optimizer=AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1),
        schedule=Schedule(),
        budget=Budget(updates=updates, batch=32),
        seeds=Seeds(model=0, data=1, batches=0),
        evaluate=Evaluate(every=every, batch=32),
        execution=CudaGraph())


def suite(task, length=16, min_length=8, ood=(32, 64)):
    return run(Synthetic(task=task, length=length, min_length=min_length, train=128, validation=32, test=64,
                         ood=ood, study=study))


recall = {"symbols": 64, "pairs": 4, "queries": 2}
numbers = {"symbols": 64, "number_limit": 128}
tasks = {
    "mqar": suite(MQAR(**recall), length=24, min_length=24, ood=(48, 96)),
    "lookup": suite(Lookup(**recall), length=24, min_length=24, ood=(48, 96)),
    "dyck": suite(Dyck(), min_length=12),
    "blocks": suite(AlternatingBlocks()),
    "histogram-bos": suite(Histogram(**numbers)),
    "histogram-no-bos": suite(Histogram(bos=False, **numbers)),
    "histogram2": suite(DoubleHistogram(**numbers)),
    "mode-none": suite(Mode(**numbers)),
    "mode-counts": suite(Mode(scratchpad="counts", **numbers)),
    "mode-itemized": suite(Mode(scratchpad="itemized", **numbers)),
    "most_freq": suite(MostFrequent(symbols=64)),
    **{f"{name}-{kind}": suite(task(symbols=64, unique=unique))
       for name, task in (("copy", Copy), ("reverse", Reverse), ("sort", Sort))
       for kind, unique in (("repeat", False), ("unique", True))},
    "dyck-2": suite(TypedDyck(), min_length=12),
    "dyck-3": suite(TypedDyck(types=3), min_length=12),
    "count": suite(Count(**numbers)),
    **{f"addition-{order}-{plain}-{carries}": suite(Addition(order=order, hints=hints, carries=carries, **numbers),
                                                    length=4, min_length=2, ood=(8, 16))
       for order in ("forward", "reverse") for plain, hints in (("plain", False), ("hints", True))
       for carries in ("standard", "balanced")},
    "parity-none": suite(Parity(**numbers)),
    "parity-running": suite(Parity(scratchpad="running", **numbers)),
    "parity-running-hints": suite(Parity(scratchpad="running", hints=True, **numbers)),
    "parity-ones-hints": suite(Parity(scratchpad="ones", hints=True, **numbers)),
    "boolean-and-shift": suite(BooleanAnd(symbols=64)),
    "boolean-and-random": suite(BooleanAnd(shift=False, symbols=64)),
    **{f"crasp-depth-{depth}": suite(CRASP(depth=depth)) for depth in (1, 2, 3)},
}

# Disjoint pools of 64 training rows.
pools = {"train": 64, "validation": 64, "test": 128, "ood": (16, 32)}
disjoint = swap(study, "disjoint", True)
parity = Parity(symbols=8, number_limit=64)
transitions = {
    "copy-repeat": run(Synthetic(task=Copy(symbols=8), length=8, min_length=4, study=disjoint, **pools), 5000, 250),
    "parity-none": run(Synthetic(task=parity, length=8, study=disjoint, **pools), 5000, 250),
    "parity-running": run(Synthetic(task=swap(parity, "scratchpad", "running"), length=8, study=disjoint, **pools),
                          5000, 250),
}

noisy = Memorization(disjoint=True, noise=0.2, noise_seed=2, tolerance=0.02)
capacity = grid(run(Synthetic(task=parity, length=8, study=noisy, **pools)),
                {"model.width": {f"width{width}": width for width in (16, 32, 64, 128)}})

control = run(Synthetic(task=RandomLM(symbols=4, number_limit=64), length=8, train=8, validation=64, test=128,
                        ood=(16, 32), study=study))

seeds = {f"seed{seed}": Seeds(model=seed, data=1, batches=seed) for seed in (0, 1, 2)}
experiments = {
    **{f"suite-{name}": experiment for name, experiment in tasks.items()},
    **grid({f"transitions-{name}": experiment for name, experiment in transitions.items()}, {"seeds": seeds}),
    **grid({f"capacity-{name}": experiment for name, experiment in capacity.items()}, {"seeds": seeds}),
    **grid({"control": control}, {"seeds": seeds}),
}
