"""The optimizers reproduce the historical optimizer zoo on the CPU.

Golden records: the `runs` of `fixtures/legacy_zoo.json`, written by the
`train_one` of the all24 benchmark and of the AMSGrad extensions with the step
uncompiled, for every recipe on GPTMini of width 32, at the top rate of its
grid, or the middle one where the top rate diverges within 40 updates.
The pinned trainers are replayed on the current backend for numerical comparisons;
the archived observations and configurations remain intact.

A rule reproduces its run bit for bit when it computes as the historical step
did, the coefficients that change with the update included: both compute them
from a float64 update count. Constant first moments and the extensions' decay
use different operation orders. Their equations are compared with the frozen
optimizers on shared inputs and gradients in both float32 and float64, within
64 machine epsilons. This measures update arithmetic without amplifying rounding
differences through a training trajectory. Under the historical first-moment
arithmetic the complete Adam, AMSGrad and MAGMA runs still agree bit for bit.
Stage reports, MAGMA masks and scales, and AMSGradMD factorizations are also
compared with the frozen trainers.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from lab.domain.spec import swap
from lab.domain.training import Checkpoint
from lab.dsl import (EVD, SGD, AdaFisher, AdaGrad, Adam, AdamNC, AdamW, AdamX, AMSGradMD, AMSGradW, Chebyshev,
                     CoupledNewton, Dash, Geometric, Guarded, Inverse, InverseSqrt, Magma, Muon, NewtonDB, RMSProp)
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.legacy import rename
from lab.infrastructure.optim import build_optimizer, coordinate
from lab.infrastructure.store import RunDirectory

from test_engine import Interrupted, train
from test_text import historical, reproduces
from reference_fixtures import runtime_fixture, runtime_run

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_zoo.json").read_text())
BETAS = (0.9, 0.999)
EXACT = {
    "sgd": lambda rate: SGD(lr=rate),
    "adagrad": lambda rate: AdaGrad(lr=rate),
    "amsgrad_inverse": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0,
                                             beta1_decay=Inverse(), lr_decay=InverseSqrt()),
    "amsgrad_geometric": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0,
                                               beta1_decay=Geometric(0.99), lr_decay=InverseSqrt()),
    "adamx": lambda rate: AdamX(lr=rate),
    "adamnc": lambda rate: AdamNC(lr=rate),
    "rmsprop": lambda rate: RMSProp(lr=rate),
    "muon": lambda rate: Muon(lr=rate),
    "muon_guarded": lambda rate: Guarded(Muon(lr=rate)),
    "dash_evd": lambda rate: Dash(lr=rate, solver=EVD()),
    "dash_ndb": lambda rate: Dash(lr=rate, solver=NewtonDB()),
    "dash_cn": lambda rate: Dash(lr=rate, solver=CoupledNewton()),
    "dash_chebyshev": lambda rate: Dash(lr=rate, solver=Chebyshev()),
    "dash_ndb_guarded": lambda rate: Guarded(Dash(lr=rate)),
    "adafisher": lambda rate: AdaFisher(lr=rate),
    "adafisherw": lambda rate: AdaFisher(lr=rate, weight_decay=0.01),
    "magma_rmsprop": lambda rate: Magma(RMSProp(lr=rate)),
    "magma_muon": lambda rate: Magma(Muon(lr=rate)),
    "magma_sgd": lambda rate: Magma(SGD(lr=rate)),
    # These sources used independent gain/auxiliary rates; new lab runs
    # schedule every component by default. Keep the source convention explicit.
    "amsgradmd": lambda rate: AMSGradMD(lr=rate, schedule_components=False),
    "amsgradmd_guarded": lambda rate: Guarded(
        AMSGradMD(lr=rate, direction_rate=3e-4, schedule_components=False), sigma=0.25),
}
CLOSE = {
    "adam": lambda rate: Adam(lr=rate),
    "amsgrad": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0),
    "adamw": lambda rate: AdamW(lr=rate, betas=BETAS, weight_decay=0.01),
    "magma_adam": lambda rate: Magma(Adam(lr=rate)),
    "magma_adamw": lambda rate: Magma(AdamW(lr=rate, betas=BETAS, weight_decay=0.01)),
    "amsgradw": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.01),
}

def recipe(name, recipes):
    golden = runtime_run("zoo", FIXTURE["runs"][name])
    return golden, swap(historical(golden), "optimizer", recipes[name](golden["config"]["rate"]))


def stages_agree(test, golden, result):
    """The guard's acceptance, MAGMA's report and the factorizations AMSGradMD ends with are the run's."""
    if golden["guard_acceptance"] is not None:
        test.assertEqual(result["optimizer"]["guard"]["acceptance"], golden["guard_acceptance"])
    if golden["magma"] is not None:
        blocks = {rename(name): block for name, block in golden["magma"]["blocks"].items()}
        test.assertEqual(result["optimizer"]["magma"], {**golden["magma"], "blocks": blocks})
    diagnostics = golden["optimizer_diagnostics"]
    if diagnostics is not None and diagnostics["md_names"]:
        matrices = {rename(name): matrix for name, matrix in diagnostics["matrices"].items()}
        test.assertEqual(result["optimizer"]["magnitude"], matrices)
        if diagnostics["accepted_steps"] is not None:
            guard = result["optimizer"]["guard"]
            test.assertEqual((guard["accepted"], guard["halved"]),
                             (diagnostics["accepted_steps"], diagnostics["halved_matrix_steps"]))


def historical_average(m, gradient, beta):
    """The first moment as `full_compile_benchmark.optimizer.coordinate` averaged it."""
    m.mul_(beta).add_(gradient * (1 - beta))


class ZooTests(unittest.TestCase):
    def test_rules_reproduce_their_runs(self):
        for name in EXACT:
            golden, experiment = recipe(name, EXACT)
            with self.subTest(recipe=name), tempfile.TemporaryDirectory() as root:
                result = train(experiment, root)
                reproduces(self, golden, root)
                stages_agree(self, golden, result)

    def test_a_guard_applies_the_directions_it_accepts(self):
        golden = FIXTURE["runs"]["sgd"]
        experiment = swap(historical(golden), "optimizer", Guarded(SGD(lr=golden["config"]["rate"])))
        with tempfile.TemporaryDirectory() as root:
            result = train(experiment, root)
            reproduces(self, golden, root)
            self.assertEqual(result["optimizer"], {"guard": {"updates": 40, "accepted": 40, "acceptance": 1.0}})

    def test_an_eigendecomposition_is_refused_a_graph(self):
        model = build_model(historical(FIXTURE["runs"]["dash_evd"]).model, 65, 0)
        with self.assertRaisesRegex(ValueError, "eigendecomposition"):
            build_optimizer(Dash(lr=1e-3, solver=EVD()), model, rate=torch.zeros(()))

    def test_only_a_guarded_magnitude_direction_rule_moves_its_directions_at_a_rate_of_their_own(self):
        model = build_model(historical(FIXTURE["runs"]["amsgradmd"]).model, 65, 0)
        with self.assertRaisesRegex(ValueError, "direction_rate"):
            build_optimizer(AMSGradMD(lr=0.3, direction_rate=3e-4), model)
        build_optimizer(Guarded(AMSGradMD(lr=0.3, direction_rate=3e-4), sigma=0.25), model)

    def test_an_interrupted_run_resumes_onto_the_same_records(self):
        golden, experiment = recipe("adamnc", EXACT)

        def interrupt(label, row):
            if row["step"] == 25:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(swap(experiment, "checkpoint", Checkpoint(every=10)), root, interrupt)
            run = RunDirectory(Path(root) / "run")
            self.assertEqual(run.checkpoint("cpu")["optimizer"]["state"][0]["step"].item(), 20)
            train(swap(experiment, "checkpoint", Checkpoint(every=10)), root)
            reproduces(self, golden, root)

    def test_constant_momenta_reproduce_their_runs_under_the_historical_arithmetic(self):
        for name in ("adam", "amsgrad", "magma_adam", "magma_adamw"):
            golden, experiment = recipe(name, CLOSE)
            with (self.subTest(recipe=name), tempfile.TemporaryDirectory() as root,
                  mock.patch.object(coordinate, "average", historical_average)):
                result = train(experiment, root)
                reproduces(self, golden, root)
                stages_agree(self, golden, result)

    def test_other_arithmetic_matches_updates_on_shared_gradients(self):
        references = runtime_fixture("zoo", "optimizer_steps")["optimizer_steps"]
        for name, variants in references.items():
            for dtype_name, rows in variants.items():
                dtype = getattr(torch, dtype_name)
                with self.subTest(recipe=name, dtype=dtype_name):
                    keys = tuple(rows[0]["point"])
                    model = torch.nn.Module()
                    model.weights = torch.nn.ParameterList([
                        torch.nn.Parameter(torch.tensor(rows[0]["point"][key], dtype=dtype)) for key in keys])
                    model.hidden_matrices = lambda: [model.weights[1]]
                    config = FIXTURE["runs"][name]["config"]
                    optimizer = build_optimizer(CLOSE[name](config["rate"]), model,
                                                updates=len(rows), seed=config["seed"])
                    optimizer.param_groups[0]["lr"] = config["rate"]
                    tolerance = 64 * torch.finfo(dtype).eps
                    for step, row in enumerate(rows, start=1):
                        for key, parameter in zip(keys, model.weights, strict=True):
                            parameter.data.copy_(torch.tensor(row["point"][key], dtype=dtype))
                            parameter.grad = torch.tensor(row["gradient"][key], dtype=dtype)
                        optimizer.step()
                        for key, parameter in zip(keys, model.weights, strict=True):
                            torch.testing.assert_close(parameter, torch.tensor(row["next"][key], dtype=dtype),
                                                       rtol=tolerance, atol=tolerance,
                                                       msg=lambda message: f"{name}/{dtype_name}/{step}/{key}: {message}")


if __name__ == "__main__":
    unittest.main()
