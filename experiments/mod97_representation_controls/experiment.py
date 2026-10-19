"""Disentangle input codes, output codes, loss and decay on identical division pairs.

Source: landscape_grokking_plan.md, M9.3, the scalar MLP study at 897e80be,
and the browser split/sampler port at c59d4426. Every network has two hidden
SiLU layers of width 195; README.md records the dimension/parameter changes.
"""

from lab.dsl import *

scalar = MLP(width=195, depth=2, encoding=Scalar(), activation=SiLU(), init=TorchDefault(), context=2)
base = Experiment(
    model=scalar,
    benchmark=DivisionPairs(data=LabDivision(prime=97, train_fraction=0.5), loss=SquaredAnswers()),
    optimizer=AdamW(lr=0.003, betas=(0.9, 0.98), eps=1e-8, weight_decay=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=30_000, batch=512),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=100, batch=512),
    execution=Eager(device="cuda"),
    diagnostics=Diagnostics(every=100, gradients=True),
    checkpoint=Checkpoint(every=1000),
)

onehot = swap(base, "model.encoding", OneHot())
vector_scalar = swap(base, "model", VectorReadout(base.model))
vector_onehot = swap(onehot, "model", VectorReadout(onehot.model))

experiments = grid(
    {"scalar-number-mse": base, "onehot-number-mse": onehot,
     "scalar-vector-mse": vector_scalar, "onehot-vector-mse": vector_onehot,
     "scalar-vector-ce": swap(vector_scalar, "benchmark.loss", CrossEntropyAnswers()),
     "onehot-vector-ce": swap(vector_onehot, "benchmark.loss", CrossEntropyAnswers())},
    {"optimizer.weight_decay": {"wd0": 0.0, "wd1": 1.0},
     "seeds": {f"s{seed}": Seeds(model=seed, data=seed, batches=seed) for seed in (11, 17, 23)}},
)
