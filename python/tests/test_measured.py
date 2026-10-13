"""Covered arithmetic, unchanged update trajectory and actual budget stops."""

import tempfile
from pathlib import Path
import unittest

import torch

from lab.dsl import Budget, Checkpoint, Compiled, Evaluate, FlopBudget, Measured, Parity, TensorStack, swap
from lab.infrastructure.arithmetic import Arithmetic
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.tensor import depth_step, parity_step, reference_table
from lab.domain.tasks import AlternatingBlocks
from lab.infrastructure.engine.measured import MeasuredStepper
from lab.infrastructure.engine.loop import Clock
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory
from test_engine import Interrupted

from examples import gptmini
from test_synthetic_training import train
from lab.dsl import AdamW, Experiment, Schedule, Seeds, Synthetic


def small(model):
    return Experiment(model=model, benchmark=Synthetic(Parity(), length=2, min_length=1, train=5,
                       validation=2, test=2, target=.99), optimizer=AdamW(lr=1e-3, betas=(.9,.98), weight_decay=.1),
                      schedule=Schedule(), budget=Budget(updates=3, batch=2), seeds=Seeds(model=0, data=1),
                      evaluate=Evaluate(every=1, batch=2), execution=Measured(device="cpu"),
                      checkpoint=Checkpoint(every=1))


class MeasuredTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_matrix_forward_backward_and_scalar_rules_have_manual_counts(self):
        left = torch.randn(2, 3, requires_grad=True)
        right = torch.randn(3, 4, requires_grad=True)
        counter = Arithmetic()
        with counter:
            (left @ right).sum().backward()
        self.assertEqual(counter.operators["aten.mm.default"], 3 * 2 * 2 * 3 * 4)
        self.assertEqual(counter.operators["aten.sum.default"], 7)
        self.assertEqual(counter.integer, 0)
        with self.assertRaisesRegex(NotImplementedError, "Uncounted"):
            with Arithmetic():
                torch.special.erf(left)

    def test_empty_reduction_never_produces_negative_cost(self):
        counter = Arithmetic()
        with counter:
            torch.empty(0).sum()
        self.assertEqual(counter.charged, 0)

    def test_tensorized_raw_tables_match_independent_rules_in_every_cell(self):
        for task, vocab, step in ((AlternatingBlocks(blocks=4), 36, depth_step), (Parity(), 68, parity_step)):
            table = reference_table(task, vocab, torch.device("cpu"))
            expected = torch.tensor([[step(token, state) for state in range(6)] for token in range(vocab)])
            torch.testing.assert_close(table, expected)

    def test_shape_probes_cover_forward_backward_adamw_and_leave_model_unchanged(self):
        for spec in (swap(gptmini(width=64, depth=2), "context", 5), TensorStack(64, 2, 5)):
            run = small(spec)
            task = build_task(run.benchmark, 1, torch.device("cpu"), run.model)
            model = build_model(spec, task.vocab, 0)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            stepper = MeasuredStepper(run, task, model, Clock(torch.device("cpu")))
            for size in (1, 2):
                profile = stepper.profile_update(size)
                self.assertGreater(profile["charged_ops"], 0)
                self.assertEqual(set(profile["phases"]), {"forward_loss", "backward", "gradient_norm", "adamw"})
                self.assertEqual(profile["charged_ops"], sum(p["charged_ops"] for p in profile["phases"].values()))
            torch.testing.assert_close(torch.get_rng_state(), rng)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name])
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_measured_execution_preserves_the_compiled_update_and_validation_trajectory(self):
        measured = small(swap(gptmini(width=64, depth=2), "context", 5))
        plain = swap(measured, "execution", Compiled(device="cpu"))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            train(plain, a)
            train(measured, b)
            first, second = RunDirectory(Path(a) / "run"), RunDirectory(Path(b) / "run")
            for key, value in first.checkpoint("cpu")["model"].items():
                torch.testing.assert_close(value, second.checkpoint("cpu")["model"][key], rtol=0, atol=0)
            for left, right in zip(first.records("history"), second.records("history"), strict=True):
                self.assertEqual(left["validation"], right["validation"])
            profiles = second.result()["arithmetic"]["training_profiles"]
            expected = 2 * profiles["2"]["charged_ops"] + profiles["1"]["charged_ops"]
            self.assertEqual(second.result()["arithmetic"]["training_charged_ops"], expected)

    def test_exact_ceiling_stops_before_an_unaffordable_update_and_preserves_sampler(self):
        run = small(TensorStack(64, 2, 5))
        task = build_task(run.benchmark, 1, torch.device("cpu"), run.model)
        model = build_model(run.model, task.vocab, 0)
        stepper = MeasuredStepper(run, task, model, Clock(torch.device("cpu")))
        cost = stepper.profile_update(2)["charged_ops"]
        cap = task.label_arithmetic["charged_ops"] + cost
        run = swap(run, "budget", FlopBudget(updates=5, batch=2, flops=cap))
        with tempfile.TemporaryDirectory() as root:
            train(run, root)
            stored = RunDirectory(Path(root) / "run")
            self.assertEqual(stored.result()["stop"], {"step": 1, "reason": "flop_budget"})
            self.assertEqual(stored.result()["arithmetic"]["training_charged_ops"], cap)
            self.assertEqual(stored.result()["arithmetic"]["unspent"], 0)
            self.assertEqual(stored.checkpoint("cpu")["cursor"], 2)

    def test_preprocessing_cannot_produce_a_result_that_already_exceeds_the_ceiling(self):
        run = small(TensorStack(64, 2, 5))
        task = build_task(run.benchmark, 1, torch.device("cpu"), run.model)
        cap = task.label_arithmetic["charged_ops"] - 1
        run = swap(run, "budget", FlopBudget(updates=5, batch=2, flops=cap))
        model = build_model(run.model, task.vocab, 0)
        with self.assertRaisesRegex(ValueError, "preparation exceeds"):
            MeasuredStepper(run, task, model, Clock(torch.device("cpu")))

    def test_checkpoint_restores_update_counts_and_charges_repeated_auxiliary_preprocessing(self):
        run = small(TensorStack(64, 2, 5))
        def interrupt(label, row):
            if row["step"] == 2:
                raise Interrupted
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed:
            train(run, full)
            with self.assertRaises(Interrupted):
                train(run, resumed, interrupt)
            train(run, resumed)
            a, b = RunDirectory(Path(full) / "run"), RunDirectory(Path(resumed) / "run")
            for name, value in a.checkpoint("cpu")["model"].items():
                torch.testing.assert_close(value, b.checkpoint("cpu")["model"][name], rtol=0, atol=0)
            ca, cb = a.result()["arithmetic"], b.result()["arithmetic"]
            self.assertEqual(ca["training_floating_ops"], cb["training_floating_ops"])
            self.assertEqual(ca["training_integer_ops"], cb["training_integer_ops"])
            self.assertEqual(cb["auxiliary_labels"], 2 * ca["auxiliary_labels"])
            self.assertEqual(cb["training_charged_ops"] - ca["training_charged_ops"], ca["auxiliary_labels"])


if __name__ == "__main__":
    unittest.main()
