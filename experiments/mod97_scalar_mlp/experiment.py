"""A perceptron reading x and y as two numbers, fitted to the whole x / y mod 97 table.

`README.md` beside this file describes the runs and what they found.
"""

from lab.dsl import *

scalar = MLP(width=256, depth=2, encoding=Scalar(), activation=GELU(), init=TorchDefault(), context=2)

# The 9,312 rows x in [0, 97), y in [1, 97) form one batch: every update is a
# full gradient step. Gradient descent steps at about half the inverse of the
# largest Hessian eigenvalue at the start: 8.02 for this perceptron.
base = Experiment(
    model=scalar,
    benchmark=ModularRegression(prime=97, operation="/"),
    optimizer=SGD(lr=1 / 16),
    schedule=Schedule(),
    budget=Budget(updates=200_000, batch=9312),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=1000, batch=9312),
    execution=Eager(),
    checkpoint=Checkpoint(every=10_000))

# Largest initial Hessian eigenvalues 2.38 (one-hot), 3.50 (width 64), 21.2 (width 1024).
onehot = swap(swap(base, "model.encoding", OneHot()), "optimizer.lr", 0.2)
adam = AdamW(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)
operations = {"benchmark.operation": {"div": "/", "add": "+"}}

experiments = {
    **grid({"gd-scalar": base, "gd-onehot": onehot}, operations),
    **grid({"adam-scalar": swap(base, "optimizer", adam), "adam-onehot": swap(onehot, "optimizer", adam)},
           operations),
    "gd-scalar-div-w64": swap(swap(base, "model.width", 64), "optimizer.lr", 0.14),
    "gd-scalar-div-w1024": swap(swap(base, "model.width", 1024), "optimizer.lr", 0.024),
}
