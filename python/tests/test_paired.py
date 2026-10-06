"""Check actual causal key/value binding, pricing, training and continuation.

Source: pairedHeadOutput and pairedBinding_fit in PairedMatchingWitness.lean.
The binding examples expose the content-only permutation invariance; they
test the model's response, not a replacement attention target. The numerical
search test asserts recovery on these observations, not global head pricing.
"""

from copy import deepcopy
import unittest

import torch

from lab.domain.atomic import (AtomicColumns, AtomicMatching, BindingPricing, MatchingBindings,
                               PairedMatching, SearchPricing)
from lab.domain.experiment import Experiment
from lab.domain.optimizers import AdamW
from lab.domain.spec import swap
from lab.domain.training import Budget, Eager, Evaluate, Schedule, Seeds
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.atomic import (MatchingHead, head_forward, paired_head_forward,
                                          paired_head_scores)
from lab.infrastructure.optim.atomic import AtomicColumns as Columns, binding_price


def binding_experiment():
    return Experiment(
        model=PairedMatching(context=6, width=4, heads=3, cap=1.0, channels=1,
                             initial_value=0.5, precision="float64"),
        benchmark=MatchingBindings(), optimizer=AtomicColumns(pricing=BindingPricing()),
        schedule=Schedule(), budget=Budget(updates=6, batch=2), seeds=Seeds(),
        evaluate=Evaluate(every=1, batch=2), execution=Eager(device="cpu"))


class PairedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.experiment = binding_experiment()
        self.task = build_task(self.experiment.benchmark, 0, torch.device("cpu"))
        self.model = build_model(self.experiment.model, self.task.vocab, 0)
        self.batch = torch.tensor([0, 1])

    def test_original_content_heads_cannot_distinguish_the_binding_swap(self):
        generator = torch.Generator().manual_seed(12)
        for _ in range(12):
            q, k = (torch.randn(5, 4, dtype=torch.float64, generator=generator) for _ in range(2))
            values = torch.randn(5, 1, dtype=torch.float64, generator=generator)
            response = head_forward(q, k, values, self.task.tokens)[:, -1]
            torch.testing.assert_close(response[0], response[1], rtol=0, atol=1e-13)
            error = self.task.loss(response[:, None], self.task.targets)
            self.assertGreaterEqual(float(error), 0.5 - 1e-13)

    def test_original_values_follow_their_neighboring_learned_key(self):
        q, k, values = (torch.zeros_like(t[0]) for t in (self.model.q, self.model.k, self.model.values))
        q[1, 2] = k[1, 2] = values[4, 0] = 1
        scores = paired_head_scores(q, k, self.task.tokens)[:, -1]
        expected = torch.zeros_like(scores)
        expected[:, 2] = 1
        torch.testing.assert_close(scores, expected, rtol=0, atol=0)
        head = MatchingHead(q, k, values, forward_map=paired_head_forward)
        response, targets = self.task.forward(head, self.batch)
        torch.testing.assert_close(response, targets, rtol=0, atol=0)
        changed = self.task.tokens.clone()
        changed[:, -1] = 2
        torch.testing.assert_close(head(self.task.tokens)[:, :5], head(changed)[:, :5], rtol=0, atol=0)

    def test_both_key_roles_receive_gradients_and_the_old_head_embeds_exactly(self):
        generator = torch.Generator().manual_seed(13)
        q, k = (torch.randn(5, 4, dtype=torch.float64, generator=generator) * 0.1 for _ in range(2))
        values = torch.randn(5, 1, dtype=torch.float64, generator=generator)
        head = MatchingHead(q, k, values, forward_map=paired_head_forward)
        self.task.loss(*self.task.forward(head, self.batch)).backward()
        for table in (head.q, head.k):
            self.assertGreater(float(table.grad[:, :2].abs().sum()), 0)
            self.assertGreater(float(table.grad[:, 2:].abs().sum()), 0)
        self.assertGreater(float(head.values.grad.abs().sum()), 0)
        q[:, 2:] = k[:, 2:] = 0
        paired = paired_head_forward(q, k, values, self.task.tokens)
        original = head_forward(q[:, :2], k[:, :2], values, self.task.tokens)
        torch.testing.assert_close(paired, original, rtol=0, atol=0)

    def test_binding_price_attains_the_global_box_bound_in_both_batch_orders(self):
        generator = torch.Generator().manual_seed(14)
        for batch in (self.batch, self.batch.flip(0)):
            for _ in range(12):
                gradient = torch.randn(2, 1, 1, dtype=torch.float64, generator=generator)
                head = binding_price(gradient, self.task.tokens[batch][:, 2], self.model)
                response = self.task.forward(head, batch)[0]
                torch.testing.assert_close((gradient * response).sum(), -gradient.abs().sum(),
                                           rtol=0, atol=1e-14)
                self.assertLessEqual(max(float(t.detach().abs().max()) for t in head.parameters()), 1)

    def test_columns_remove_the_binding_floor_with_a_global_certificate(self):
        solver = Columns(self.experiment.optimizer, self.model, self.task, 0)
        self.assertAlmostEqual(float(self.task.loss(*self.task.forward(self.model, self.batch)).detach()), 0.5)
        for _ in range(3):
            loss = solver.step(self.batch)
        self.assertLess(loss, 1e-24)
        self.assertLess(solver.report()["global_gap_bound_after"], 1e-12)
        self.assertTrue(solver.report()["pricing_is_global"])
        self.assertLessEqual(int(self.model.active), 3)

    def test_paired_search_and_selected_heads_resume_with_the_same_encoder(self):
        optimizer = AtomicColumns(pricing=SearchPricing(restarts=2, steps=8), correction_steps=32)
        model = build_model(swap(self.experiment.model, "heads", 9), 5, 0)
        solver = Columns(optimizer, model, self.task, 15)
        solver.step(self.batch)
        checkpoint, state = deepcopy(solver.state_dict()), deepcopy(model.state_dict())
        solver.step(self.batch.flip(0))
        expected, report = deepcopy(model.state_dict()), solver.report()
        restored = build_model(model.spec, 5, 999)
        restored.load_state_dict(state)
        continued = Columns(optimizer, restored, self.task, 999)
        continued.load_state_dict(checkpoint)
        continued.step(self.batch.flip(0))
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.assertEqual(report, continued.report())
        self.assertIs(restored.forward_map, paired_head_forward)
        self.assertFalse(report["pricing_is_global"])
        self.assertLessEqual(restored.inspect()["maximum_coordinate"], 1)

    def test_analytic_binding_pricing_requires_its_actual_encoder_and_two_observations(self):
        with self.assertRaisesRegex(ValueError, "even width"):
            swap(self.experiment.model, "width", 3)
        ordinary = AtomicMatching(context=6, width=4, heads=3, cap=1, channels=1)
        with self.assertRaisesRegex(ValueError, "scalar paired"):
            swap(self.experiment, "model", ordinary)
        with self.assertRaisesRegex(ValueError, "both observations"):
            swap(self.experiment, "budget.batch", 1)
        raw = swap(self.experiment, "optimizer", AdamW(lr=0.01, betas=(0.9, 0.999), weight_decay=0))
        self.assertIsInstance(raw.model, PairedMatching)


if __name__ == "__main__":
    unittest.main()
