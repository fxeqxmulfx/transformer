"""Frozen linear weights need gradients through them, without Fisher factor storage."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

import lean_optim as opt


class FrozenFisherTests(unittest.TestCase):
    """Exclude frozen weights from collection without breaking upstream learning."""

    def test_frozen_middle_layer_has_no_hooks_or_buffers_and_keeps_upstream_gradients(self) -> None:
        """Uniform first inputs/last derivatives make active updates equal SGD / damping."""
        model = nn.Sequential(
            nn.Linear(3, 4, bias=False, dtype=torch.float64),
            nn.Linear(4, 4, bias=False, dtype=torch.float64),
            nn.Linear(4, 2, bias=False, dtype=torch.float64),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)
        frozen = model[1].weight.requires_grad_(requires_grad=False)
        control = copy.deepcopy(model)
        optimizer = opt.AdaFisher(model, lr=0.03125, beta=0.0, gamma=1.0, damping=0.25)
        expected = opt.standard.SGD(control.parameters(), lr=0.125)
        try:
            self.assertFalse(model[1]._forward_hooks)  # noqa: SLF001 - Check that frozen layers never register collectors.
            inputs = torch.ones(2, 3, dtype=torch.float64)
            for _ in range(2):
                optimizer.zero_grad()
                expected.zero_grad()
                model(inputs).sum().backward()
                control(inputs).sum().backward()
                self.assertIsNone(frozen.grad)
                self.assertGreater(float(model[0].weight.grad.norm()), 0)
                for parameter, reference in zip(
                    model.parameters(), control.parameters(), strict=True
                ):
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
                self.assertNotIn(frozen, optimizer.factors.fresh)
                self.assertEqual(
                    set(optimizer.factors.fresh), set(optimizer.param_groups[0]["params"])
                )
                self.assertTrue(
                    all(
                        frozen not in workspace.buffers
                        for workspace in optimizer.factors.workspace.values()
                    )
                )
                optimizer.step()
                expected.step()
                for parameter, reference in zip(
                    model.parameters(), control.parameters(), strict=True
                ):
                    torch.testing.assert_close(parameter, reference, rtol=1e-13, atol=1e-13)
        finally:
            optimizer.close()

    def test_weight_frozen_after_construction_does_not_collect_factors(self) -> None:
        """An installed collector must skip a newly frozen weight before allocating buffers."""
        model = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        optimizer = opt.AdaFisher(model)
        model.weight.requires_grad_(requires_grad=False)
        inputs = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
        try:
            model(inputs).sum().backward()
            torch.testing.assert_close(inputs.grad, model.weight.detach().sum(0).expand_as(inputs))
            self.assertFalse(optimizer.factors.fresh)
            self.assertFalse(optimizer.factors.workspace)
            optimizer.step()
        finally:
            optimizer.close()


if __name__ == "__main__":
    unittest.main()
