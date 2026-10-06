"""ANSR source fidelity, fresh-batch comparisons, tied weights and resumable training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from lab.application.report import report_study
from lab.application.study import Study
from lab.domain.spec import swap
from lab.dsl import (ANSR, Attention, Block, Budget, Checkpoint, Compiled, Diagnostics, Eager, Evaluate,
                     Experiment, FFN, FusedQKV, Normal, Parity, PreNorm, QKNorm, ReLU2, RMSNorm, RoPE,
                     Schedule, Seeds, Softmax, Synthetic, Tied, Transformer, XSA)
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer
from lab.infrastructure.store import RunDirectories, RunDirectory

from test_engine import train


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.linspace(-1, 1, 16))

    def forward(self):
        return (self.x - 0.3).square().mean()


def tiny_run():
    attention = Attention(heads=2, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    model = Transformer(width=8, depth=1, block=block, positions=RoPE(), readout=Tied(),
                        final_norm=RMSNorm(), init=Normal(0.02), context=64)
    return Experiment(model=model, benchmark=Synthetic(Parity(), length=4, min_length=1,
                                                      train=32, validation=8, test=8),
                      optimizer=ANSR(popsize=4, batch_size=2), schedule=Schedule(),
                      budget=Budget(updates=4, batch=4), seeds=Seeds(), evaluate=Evaluate(every=2, batch=8),
                      execution=Eager(device="cpu"), diagnostics=Diagnostics(every=2),
                      checkpoint=Checkpoint(every=2))


class ANSRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fixed_objective_matches_original_source_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "ansr_original.json").read_text())
        model = Quadratic()
        optimizer = build_optimizer(ANSR(popsize=8, batch_size=8, bound=5), model, seed=42)
        for expected in fixture["steps"]:
            loss = optimizer.step(lambda candidate: candidate())
            self.assertEqual(float(loss), expected["loss"])
            self.assertEqual(model.x.tolist(), expected["x"])
            self.assertEqual(optimizer.report()["restarts"], expected["restarts"])
        self.assertLess(float(loss), fixture["initial_loss"])

    def test_attractors_are_compared_on_the_current_batch(self):
        model = Quadratic()
        optimizer = build_optimizer(ANSR(popsize=8, batch_size=4, bound=5), model, seed=0)
        optimizer.step(lambda candidate: candidate())
        prior = optimizer.population["best_res"].clone()
        optimizer.step(lambda candidate: candidate() + 100)
        state = optimizer.population
        self.assertTrue(torch.all(state["best_res"] >= 100))
        self.assertTrue(torch.all(prior < 100))
        self.assertEqual((state["nfev"], state["refresh_nfev"]), (24, 8))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_resume_preserves_population_and_generator(self):
        model = Quadratic()
        spec = ANSR(popsize=8, batch_size=4, bound=5)
        optimizer = build_optimizer(spec, model, seed=42)
        for _ in range(3):
            optimizer.step(lambda candidate: candidate())
        saved, parameters = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(model.state_dict())
        continued = Quadratic()
        continued.load_state_dict(parameters)
        resumed = build_optimizer(spec, continued, seed=999)
        resumed.load_state_dict(saved)
        for _ in range(5):
            self.assertEqual(float(optimizer.step(lambda candidate: candidate())),
                             float(resumed.step(lambda candidate: candidate())))
            self.assertTrue(torch.equal(model.x, continued.x))
        for key in ("pos", "best_pos", "best_res"):
            self.assertTrue(torch.equal(optimizer.population[key], resumed.population[key]))
        self.assertEqual(optimizer.report(), resumed.report())

    def test_fitness_collisions_restart_losers_and_preserve_winner(self):
        model = Quadratic()
        optimizer = build_optimizer(ANSR(popsize=8, batch_size=4, bound=5), model, seed=0)
        optimizer.step(lambda candidate: candidate() * 0 + 1)
        self.assertEqual(optimizer.report()["restarts"], 7)
        self.assertTrue(torch.isfinite(optimizer.population["best_res"][0]))
        optimizer.step(lambda candidate: candidate() * 0 + 1)
        self.assertEqual(optimizer.report()["restarts"], 14)

    def test_functional_population_preserves_the_tied_readout(self):
        run = tiny_run()
        model = build_model(run.model, vocab=8, seed=0)
        optimizer = build_optimizer(run.optimizer, model, seed=0)
        optimizer.initialize()
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            values = optimizer.evaluate(optimizer.population["pos"], lambda candidate: candidate(tokens).square().mean())
            for i, unit in enumerate(optimizer.population["pos"]):
                optimizer.install(unit)
                self.assertTrue(torch.allclose(values[i], model(tokens).square().mean(), atol=1e-5, rtol=1e-5))
        self.assertIs(model.embed.weight, model.readout.weight)

    def test_invalid_population_compositions_fail_before_training(self):
        run = tiny_run()
        for spec in (dict(popsize=1), dict(p_self=-.1), dict(bound=0), dict(batch_size=65), dict(sigma=float("nan"))):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                ANSR(**spec)
        for path, value in (("schedule", Schedule(warmup=1)), ("execution", Compiled()),
                            ("diagnostics", Diagnostics(gradients=True))):
            with self.subTest(path=path), self.assertRaises(ValueError):
                swap(run, path, value)

    def test_end_to_end_training_reports_calls_without_gradient_norms(self):
        experiment = tiny_run()
        with tempfile.TemporaryDirectory() as root:
            result = train(experiment, root)
            directory = RunDirectory(Path(root) / "run")
            self.assertEqual(directory.records("gradients"), [])
            self.assertEqual(result["optimizer"]["updates"], 4)
            self.assertGreater(result["optimizer"]["refresh_nfev"], 0)
            self.assertGreater(result["optimizer"]["nfev"], 4 * experiment.optimizer.popsize)
            study = Study("ansr", "# ANSR test\n", {"run": experiment})
            report = report_study(study, [], RunDirectories(root))
            self.assertEqual(report["runs"]["run"]["status"], "finished")
            self.assertEqual(report["rate_selection"], [])


if __name__ == "__main__":
    unittest.main()
