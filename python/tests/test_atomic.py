"""Check the numerical physical model, certificates, real training and resume.

The exact observations and error floors come from AtomicMatchingContext.lean
at 5d91bc4. The global box bound comes from matchingHeadOutput_bounds.
Approximate head-search tests assert feasibility and reproducibility, not
global optimality of its nonconvex Q/K search.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from lab.application.study import Study, run_study
from lab.domain.atomic import AtomicColumns, AtomicMatching, MatchingOrders, OrderPricing, SearchPricing
from lab.domain.experiment import Experiment
from lab.domain.generative import Parity
from lab.domain.spec import swap
from lab.domain.synthetic import Synthetic
from lab.domain.training import Budget, Checkpoint, Compiled, Diagnostics, Eager, Evaluate, Schedule, Seeds
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine import Engine
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.atomic import MatchingHead, head_forward
from lab.infrastructure.optim.atomic import AtomicColumns as Columns, order_price, simplex_squared_fit
from lab.infrastructure.store import RunDirectories, RunDirectory


def order_experiment():
    return Experiment(
        model=AtomicMatching(context=2, width=1, heads=3, cap=1.0, channels=1,
                             initial_value=0.75, precision="float64"),
        benchmark=MatchingOrders(), optimizer=AtomicColumns(pricing=OrderPricing()), schedule=Schedule(),
        budget=Budget(updates=6, batch=2), seeds=Seeds(), evaluate=Evaluate(every=1, batch=2),
        execution=Eager(device="cpu"), diagnostics=Diagnostics(every=1), checkpoint=Checkpoint(every=1))


def train(experiment, root, progress=lambda label, row: None):
    return run_study(Study("atomic", "Numerical atomic check", {"run": experiment}), [],
                     RunDirectories(root), Engine(), progress)["run"]


class AtomicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.experiment = order_experiment()
        self.task = build_task(self.experiment.benchmark, 0, torch.device("cpu"))
        self.model = build_model(self.experiment.model, self.task.vocab, 0)
        self.batch = torch.tensor([0, 1])

    def test_physical_forward_matches_lean_order_head_and_is_causal(self):
        q = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        head = MatchingHead(q, q, q)
        output, _ = self.task.forward(head, self.batch)
        torch.testing.assert_close(output.flatten(), q.new_tensor([1.0, 0.5]), rtol=0, atol=0)
        tokens = torch.tensor([[0, 1]])
        changed = torch.tensor([[0, 0]])
        torch.testing.assert_close(head(tokens)[:, 0], head(changed)[:, 0], rtol=0, atol=0)

    def test_actual_probability_mixture_is_affine_and_keeps_independent_values(self):
        q = self.model.q[0].detach().clone()
        head = MatchingHead(q.new_tensor([[0.0], [1.0]]), q.new_tensor([[0.0], [1.0]]),
                            q.new_tensor([[0.0], [1.0]]))
        first = self.task.forward(self.model, self.batch)[0].detach()
        second = self.task.forward(head, self.batch)[0].detach()
        self.model.append(head)
        with torch.no_grad():
            self.model.mass[:2].copy_(q.new_tensor([0.3, 0.7]))
        actual = self.task.forward(self.model, self.batch)[0]
        torch.testing.assert_close(actual, 0.3 * first + 0.7 * second, rtol=0, atol=1e-15)
        torch.testing.assert_close(self.model.values[1], head.values, rtol=0, atol=0)

    def test_analytic_price_attains_the_full_family_lower_bound_in_either_order(self):
        generator = torch.Generator().manual_seed(8)
        for order in (torch.tensor([0, 1]), torch.tensor([1, 0])):
            final_tokens = self.task.tokens[order][:, -1]
            for _ in range(12):
                gradient = torch.randn(2, 1, 1, dtype=torch.float64, generator=generator)
                head = order_price(gradient, final_tokens, self.model)
                response = self.task.forward(head, order)[0]
                price = (gradient * response).sum()
                torch.testing.assert_close(price, -gradient.abs().sum(), rtol=0, atol=1e-14)
                random = [torch.rand(2, 1, generator=generator, dtype=torch.float64) * 2 - 1
                          for _ in range(3)]
                random_price = (gradient * self.task.forward(MatchingHead(*random), order)[0]).sum()
                self.assertGreaterEqual(float(random_price.detach()), float(price.detach()) - 1e-14)

    def test_saturated_physical_head_is_inside_a_flat_sparsemax_region(self):
        spec = swap(swap(self.experiment.model, "initial_query", 1.0), "initial_key_spread", 1.0)
        model = build_model(spec, 2, 0)
        output, targets = self.task.forward(model, self.batch)
        loss = self.task.loss(output, targets)
        self.assertEqual(float(loss.detach()), 1 / 8)
        loss.backward()
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
        perturbed = model.q.detach()[0] * 0.9
        actual = head_forward(perturbed, model.k[0], model.values[0], self.task.tokens)
        torch.testing.assert_close(actual[:, 1], output[:, 0], rtol=0, atol=0)

    def test_columns_escape_both_flat_initializations_and_certify_zero(self):
        for saturated in (False, True):
            spec = self.experiment.model
            if saturated:
                spec = swap(swap(spec, "initial_query", 1.0), "initial_key_spread", 1.0)
            model = build_model(spec, 2, 0)
            solver = Columns(self.experiment.optimizer, model, self.task, 0)
            previous = 1 / 8
            for _ in range(3):
                loss = solver.step(self.batch)
                self.assertLessEqual(loss, previous + 1e-14)
                previous = loss
            self.assertLess(loss, 1e-24)
            self.assertLess(solver.report()["global_gap_bound_after"], 1e-12)
            self.assertTrue(solver.report()["pricing_is_global"])
            self.assertGreaterEqual(float(model.mass.min()), 0)
            self.assertAlmostEqual(float(model.mass.sum()), 1)
            self.assertLessEqual(model.inspect()["maximum_coordinate"], 1)
            self.assertLessEqual(int(model.active), 3)

    def test_unattainable_answers_reach_positive_minimum_with_zero_gap(self):
        task = build_task(MatchingOrders(targets=(2.0, -2.0)), 0, torch.device("cpu"))
        solver = Columns(self.experiment.optimizer, self.model, task, 0)
        self.assertAlmostEqual(solver.step(self.batch), 2)
        self.assertLess(solver.report()["global_gap_bound_after"], 1e-12)

    def test_small_qp_handles_duplicate_columns_and_boundary_optima(self):
        columns = torch.tensor([[0.75, 0.75], [0.75, 0.75], [1.0, -1.0]], dtype=torch.float64)
        targets = torch.tensor([2.0, -2.0], dtype=torch.float64)
        weights = simplex_squared_fit(columns, targets)
        torch.testing.assert_close(weights, weights.new_tensor([0, 0, 1]), rtol=0, atol=1e-14)

    def test_search_prices_original_values_and_checkpoints_rng_and_selected_heads(self):
        optimizer = AtomicColumns(pricing=SearchPricing(restarts=2, steps=3), correction_steps=8)
        model = build_model(swap(self.experiment.model, "heads", 5), 2, 0)
        solver = Columns(optimizer, model, self.task, 12)
        solver.step(self.batch)
        checkpoint, model_state = deepcopy(solver.state_dict()), deepcopy(model.state_dict())
        solver.step(self.batch.flip(0))
        expected_model, expected_report = deepcopy(model.state_dict()), solver.report()
        restored = build_model(model.spec, 2, 123)
        restored.load_state_dict(model_state)
        continued = Columns(optimizer, restored, self.task, 999)
        continued.load_state_dict(checkpoint)
        continued.step(self.batch.flip(0))
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
        self.assertEqual(continued.report(), expected_report)
        torch.testing.assert_close(continued.generator.get_state(), solver.generator.get_state(), rtol=0, atol=0)
        self.assertFalse(continued.report()["pricing_is_global"])
        self.assertGreater(continued.report()["search_evaluations"], 0)
        self.assertLessEqual(restored.inspect()["maximum_coordinate"], 1)

    def test_end_to_end_order_training_and_budget_extension_preserve_the_selected_optimum(self):
        with tempfile.TemporaryDirectory() as root:
            first = train(swap(self.experiment, "budget.updates", 3), root)
            self.assertEqual(first["best"]["step"], 2)
            self.assertLess(first["best"]["validation"]["loss"], 1e-24)
            extended = train(self.experiment, root)
            self.assertEqual(extended["stop"], {"step": 6, "reason": "budget"})
            self.assertEqual(extended["best"], first["best"])
            self.assertEqual(RunDirectory(Path(root) / "run").checkpoint("cpu")["step"], 6)

    def test_actual_synthetic_answer_loss_and_free_generation_work_with_generated_heads(self):
        benchmark = Synthetic(task=Parity(), length=3, train=12, validation=4, test=4, target=None)
        experiment = Experiment(
            model=AtomicMatching(context=benchmark.context, width=2, heads=3, cap=2.0),
            benchmark=benchmark, optimizer=AtomicColumns(pricing=SearchPricing(restarts=1, steps=2)),
            schedule=Schedule(), budget=Budget(updates=2, batch=4), seeds=Seeds(data=1),
            evaluate=Evaluate(every=1, batch=4), execution=Eager(device="cpu"))
        with tempfile.TemporaryDirectory() as root:
            result = train(experiment, root)
            self.assertIn("test", result["best"])
            self.assertIn("teacher_forced", result["best"]["test"])
            self.assertFalse(result["optimizer"]["pricing_is_global"])
            self.assertLessEqual(result["best"]["atomic"]["maximum_coordinate"], 2)

    def test_invalid_solver_compositions_are_rejected_at_configuration_time(self):
        with self.assertRaisesRegex(ValueError, "scalar heads"):
            swap(self.experiment, "model.width", 2)
        with self.assertRaisesRegex(ValueError, "both observations"):
            swap(self.experiment, "budget.batch", 1)
        with self.assertRaisesRegex(ValueError, "Eager"):
            swap(self.experiment, "execution", Compiled())
        with self.assertRaisesRegex(ValueError, "finite"):
            swap(self.experiment, "model.cap", float("inf"))


if __name__ == "__main__":
    unittest.main()
