"""The basis in both modes, on the two models its modes are set for, and the runs that set its recipes.

`README.md` beside this file says what the runs ask and what they found.
"""

from lab.dsl import *


def gptmini(width, depth):
    """GPTMini with four heads."""
    attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=width, depth=depth, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(0.02), context=64)


MODELS = {"small": gptmini(64, 2), "large": gptmini(128, 6)}
RATES = {"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3, "1e-2": 1e-2}


def held(run):
    """`run` with its rate held after the warmup."""
    return swap(run, "schedule", Schedule(warmup=50))


def judged128(run):
    """`run` judged at length 128, as the hard mode judges its depth."""
    return swap(run, "benchmark.select", "length-128")


def recipe(run, batch, rate, updates):
    """`run` at `batch` rows per update and `rate` for `updates` updates, observed after every 3,200 examples."""
    run = swap(swap(run, "budget", Budget(updates=updates, batch=batch)), "evaluate.every", 3_200 // batch)
    return swap(run, "optimizer.lr", RATES[rate])


def sweep(run, batch, rate):
    """`run` at `batch` rows per update and `rate` held, over 192,000 examples."""
    return recipe(held(run), batch, rate, 192_000 // batch)


def parity(run, n, updates):
    """`run` on parity of 1 to `n` bits, through a context of their size, for `updates` updates."""
    benchmark = swap(run.benchmark, "length", n)
    wide = swap(run, "model.context", max(run.model.context, benchmark.context))
    run = swap(swap(wide, "benchmark", benchmark), "model.context", benchmark.context)
    return swap(run, "budget.updates", updates)


def timed(run, batch):
    """`run` for 300 updates at `batch` rows per update and its rate held, observed after every 100."""
    run = swap(swap(held(run), "budget", Budget(updates=300, batch=batch)), "evaluate.every", 100)
    return swap(run, "stopping", None)


# A hard parity run is its easy one.
runs = {f"{mode}-{name}-{task}-seed{seed}": run
        for mode in ("easy", "hard") for name, model in MODELS.items() for seed in (0, 1, 2)
        for task, run in basis(model, mode, seed).items() if (mode, task) != ("hard", "parity")}

# The easy tasks on the small model, depth judged at length 128, at four batches and rates.
EASY = {task: judged128(run) if task == "depth" else run for task, run in basis(MODELS["small"], "easy").items()}
easy = {f"sweep-easy-{task}-b{batch}-lr{rate}": sweep(run, batch, rate)
        for task, run in EASY.items() for batch in (16, 32, 64, 128) for rate in ("3e-4", "1e-3", "3e-3", "1e-2")}

# The hard tasks, and parity of 20 and 24 bits, at the rows per update and rates tried on each model.
TRIED = {"depth": {"large": ((8, "1e-3"), (16, "1e-3"), (16, "3e-4"), (32, "1e-3")),
                   "small": ((16, "1e-3"), (16, "3e-4"))},
         "recall": {"large": ((16, "3e-4"), (16, "1e-4"), (32, "3e-4"), (64, "1e-3")),
                    "small": ((16, "3e-4"), (16, "1e-3"))},
         "parity24": {"large": ((16, "3e-4"), (16, "1e-4"), (32, "3e-4")), "small": ((16, "3e-4"),)},
         "parity20": {"large": ((16, "3e-4"), (16, "1e-4")), "small": ((16, "3e-4"), (16, "1e-3"))}}
HARD = {name: {**basis(model, "hard"), **{f"parity{n}": parity(basis(model, "hard")["parity"], n, 16_000)
                                          for n in (20, 24)}}
        for name, model in MODELS.items()}
hard = {f"sweep-hard-{task}-{name}-b{batch}-lr{rate}": sweep(HARD[name][task], batch, rate)
        for task, tried in TRIED.items() for name, points in tried.items() for batch, rate in points}

# E_2 judged at length 128: the small model from five seeds at four recipes, the large from three at one.
DEPTH128 = {"small": ((16, "1e-3", 4_000), (16, "3e-4", 8_000), (64, "1e-3", 2_000), (128, "3e-4", 1_000)),
            "large": ((16, "1e-3", 4_000),)}
depth128 = {f"depth128-{name}-b{batch}-lr{rate}-seed{seed}":
            recipe(judged128(basis(MODELS[name], "easy", seed)["depth"]), batch, rate, updates)
            for name, points in DEPTH128.items() for batch, rate, updates in points
            for seed in range(5 if name == "small" else 3)}

# The easy recall at rate 3e-4 on the small model; at its rate annealed by a half cosine to a tenth at the budget; and
# on 100,000 training rows, Zoology's, for twice its budget.
ANNEALED = Schedule(warmup=50, anneal=Cosine(start=50, end=4_800, final=0.1))
recall = {**{f"recall-small-lr3e-4-seed{seed}":
             recipe(basis(MODELS["small"], "easy", seed)["recall"], 64, "3e-4", 6_400) for seed in (0, 1, 2)},
          **{f"recall-annealed-{name}-seed{seed}": swap(basis(model, "easy", seed)["recall"], "schedule", ANNEALED)
             for name, model in MODELS.items() for seed in (0, 1, 2)},
          **{f"recall-rows100k-{name}-seed{seed}":
             swap(swap(basis(model, "easy", seed)["recall"], "benchmark.train", 100_000), "budget.updates", 9_600)
             for name, model in MODELS.items() for seed in (0, 1, 2)}}

# Parity of 1 to 24 bits for 16,000 updates from three seeds: as the hard mode, with the rate annealed, on more rows.
VARIANTS = {"": lambda run: run,
            "anneal-": lambda run: swap(run, "schedule", Schedule(warmup=50, anneal=Cosine(start=50, end=16_000,
                                                                                        final=0.1))),
            "rows100k-": lambda run: swap(run, "benchmark.train", 100_000)}
lengths = {f"parity24-{variant}{name}-seed{seed}": change(parity(basis(model, "hard", seed)["parity"], 24, 16_000))
           for variant, change in VARIANTS.items() for name, model in MODELS.items() for seed in (0, 1, 2)}

# At 128 rows per update the large model's hard depth leaves too little of a GTX 1050's 2 GB beside its captured graphs
# to evaluate its test split at length 128.
times = {f"time-{mode}-{task}-{name}-b{batch}": timed(run, batch)
         for mode, name, batches in (("easy", "small", (16, 32, 64, 128, 256)), ("hard", "large", (16, 32, 64, 128)))
         for task, run in basis(MODELS[name], mode).items() for batch in batches
         if (mode, task, batch) != ("hard", "depth", 128)}

experiments = {**runs, **easy, **hard, **depth128, **recall, **lengths, **times}
