"""Analytic constant/reversal witnesses for the task166 read-only diagnostic."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch

from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW

source = Path(__file__).resolve().parents[2] / "experiments/basis_control/analysis/schedule_free_directions.py"
spec = importlib.util.spec_from_file_location("schedule_free_directions", source)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ScheduleFreeDirectionTests(unittest.TestCase):
    def test_constant_direction_and_reversal_have_opposite_averaging_effects(self):
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        optimizer = ResidualScheduleFreeAdamW([p], lr=.2, betas=(.9, .98), weight_decay=0.)
        for t in range(1, 21):
            p.grad = torch.ones_like(p)
            optimizer.step()
            expected = -(.55 * t + .45) * .2 / (1 + optimizer.param_groups[0]["eps"])
            self.assertAlmostEqual(p.item(), expected, places=13)
        group = optimizer.param_groups[0]
        state = optimizer.state[p]
        for sign, oppose in ((1., False), (-1., True)):
            g = torch.full_like(p, sign)
            terms = probe.directions(p, g, state, group)
            result = probe.contributions(g, terms)
            self.assertEqual(result["strong_local_opposition"], oppose)
            self.assertEqual(result["gradient_dot_total"] > 0, oppose)

    def test_diagnostic_preserves_inputs_and_predicts_a_stateful_step(self):
        p = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
        optimizer = ResidualScheduleFreeAdamW([p], lr=.03, betas=(.7, .92), weight_decay=.2)
        for gradient in ([.2, -.4], [-.3, .5], [.1, .9]):
            p.grad = torch.tensor(gradient, dtype=torch.float64)
            optimizer.step()
        p.grad = torch.tensor([.6, -.8], dtype=torch.float64)
        before, state = p.detach().clone(), copy.deepcopy(optimizer.state_dict())
        terms = probe.directions(p, p.grad, optimizer.state[p], optimizer.param_groups[0])
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        self.assertEqual(state["param_groups"], optimizer.state_dict()["param_groups"])
        for key, value in state["state"][0].items():
            torch.testing.assert_close(value, optimizer.state_dict()["state"][0][key], rtol=0, atol=0)
        optimizer.step()
        torch.testing.assert_close(p, before + sum(terms), rtol=1e-14, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
