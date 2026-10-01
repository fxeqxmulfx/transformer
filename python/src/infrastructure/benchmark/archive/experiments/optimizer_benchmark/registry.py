"""Predeclared optimizer coverage and equally sized learning-rate grids."""

from dataclasses import dataclass

from .coordinate import CoordinateOptimizer
from .fisher import AdaFisherOptimizer
from .matrix import DashOptimizer, MuonOptimizer


@dataclass(frozen=True)
class Method:
    name: str
    rates: tuple[float, float, float]
    scope: str


METHODS = (
    Method("sgd", (0.1, 0.3, 1.0), "reference baseline; SGD rule formalized"),
    Method("adagrad", (0.03, 0.1, 0.3), "AdaGrad rule and monotone metric formalized"),
    Method("adam", (0.0003, 0.001, 0.003), "paper Adam without debiasing; counterexamples formalized"),
    Method("adamw", (0.0003, 0.001, 0.003), "practical bias-corrected reference; decay 0.01"),
    Method("amsgrad", (0.0003, 0.001, 0.003), "constant-momentum deterministic convergence extension"),
    Method("amsgrad_inverse", (0.003, 0.01, 0.03), "source inverse-momentum regret and offline convergence"),
    Method("amsgrad_geometric", (0.003, 0.01, 0.03), "source geometric-momentum regret and offline convergence"),
    Method("adamx", (0.003, 0.01, 0.03), "AdamX regret bound; inverse momentum"),
    Method("adamnc", (0.01, 0.03, 0.1), "AdamNC regret bound; inverse momentum and second-moment averaging"),
    Method("muon", (0.003, 0.01, 0.03), "paper finite Newton-Schulz recipe; unguarded training can cycle"),
    Method("muon_guarded", (0.1, 0.3, 1.0), "formalized global direction safeguard"),
    Method("dash_evd", (0.0003, 0.001, 0.003), "exact regularized spectral inverse-root specification"),
    Method("dash_ndb", (0.0003, 0.001, 0.003), "corrected finite chained NDB solver"),
    Method("dash_cn", (0.0003, 0.001, 0.003), "corrected finite fourth-root coupled Newton solver"),
    Method("dash_chebyshev", (0.0003, 0.001, 0.003), "corrected cosine fit, sample guard and Clenshaw scaling"),
    Method("dash_ndb_guarded", (0.1, 0.3, 1.0), "finite NDB/grafting with formalized global training safeguard"),
    Method("adafisher", (0.00001, 0.0001, 0.001), "corrected factors, bias correction and deterministic convergence"),
    Method("adafisherw", (0.00001, 0.0001, 0.001), "formalized decoupled-decay update; general convergence not proved"),
)


def make_optimizer(name, model, lr):
    parameters = model.named_parameters()
    if name.startswith("muon"):
        return MuonOptimizer(parameters, lr, guarded=name.endswith("guarded"))
    if name.startswith("dash_"):
        solver = name.removeprefix("dash_").removesuffix("_guarded")
        return DashOptimizer(parameters, lr, solver=solver, guarded=name.endswith("guarded"))
    if name.startswith("adafisher"):
        return AdaFisherOptimizer(model, lr, decay=0.01 if name == "adafisherw" else 0.0)
    if name.startswith("amsgrad_"):
        schedule = name.removeprefix("amsgrad_")
        return CoordinateOptimizer(parameters, lr, rule="amsgrad", schedule=schedule)
    if name in {"adamx", "adamnc"}:
        return CoordinateOptimizer(parameters, lr, rule=name, schedule="inverse")
    if name in {"sgd", "adagrad", "adam", "adamw", "amsgrad"}:
        return CoordinateOptimizer(parameters, lr, rule=name, decay=0.01 if name == "adamw" else 0.0)
    raise ValueError(name)
