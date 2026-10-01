"""Fixed grids recorded in PLAN.md before Magma training."""

from experiments.optimizer_benchmark.registry import Method, make_optimizer as dense_optimizer
from .optimizer import MagmaOptimizer, RMSPropOptimizer


METHODS = (
    Method("rmsprop", (0.0001, 0.0003, 0.001), "dense raw-variance RMSProp control"),
    Method("magma_rmsprop", (0.0003, 0.001, 0.003), "Algorithm 1 with raw-variance RMSProp"),
    Method("magma_adam", (0.0009, 0.003, 0.009), "Algorithm 1 with frozen raw-moment Adam"),
    Method("magma_adamw", (0.0009, 0.003, 0.009), "Algorithm 1 with frozen bias-corrected AdamW"),
    Method("magma_muon", (0.01, 0.03, 0.1), "Algorithm 1 with frozen Muon/auxiliary-Adam hybrid"),
    Method("magma_sgd", (0.1, 0.3, 1.0), "Algorithm 1 with frozen SGD and dense scoring EMA"),
)


def make_optimizer(name, model, lr):
    base_name = name.removeprefix("magma_")
    base = RMSPropOptimizer(model.named_parameters(), lr) if base_name == "rmsprop" else dense_optimizer(base_name, model, lr)
    if name.startswith("magma_"):
        # Set by the runner from the paired model seed before constructing the optimizer.
        return MagmaOptimizer(base, seed=model.magma_seed)
    return base
