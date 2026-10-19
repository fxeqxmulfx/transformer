"""The lab schedule reaches every MD rate, including a zero first update."""

import copy
from types import SimpleNamespace
import unittest

import lean_optim
import torch

from lab.domain.optimizers import AMSGradMD, Guarded
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.optim import build_optimizer


def model():
    result = torch.nn.Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        result.weight.copy_(torch.arange(1, 7, dtype=torch.float64).view(2, 3) / 8)
        result.bias.copy_(torch.tensor([.25, -.25], dtype=torch.float64))
    result.hidden_matrices = lambda: [result.weight]
    return result


class Task:
    inputs_table = torch.tensor([[.25, -.5, .75], [-.25, .125, .5]], dtype=torch.float64)

    def inputs(self, rows):
        return self.inputs_table[rows]

    def forward(self, network, batch):
        output = network(batch)
        return output, torch.zeros_like(output)

    def loss(self, output, targets):
        return torch.nn.functional.mse_loss(output, targets)


class MDWarmupTests(unittest.TestCase):
    def test_eager_steps_follow_independently_scheduled_component_callbacks(self):
        task = Task()
        parts = ((torch.arange(2), 0, 2),)
        for guarded in (False, True):
            with self.subTest(guarded=guarded):
                source, reference = model(), model()
                base = AMSGradMD(lr=.125, gain_rate=.03125, auxiliary_rate=.015625,
                                direction_rate=.0625 if guarded else None)
                spec = Guarded(base, sigma=.5) if guarded else base
                experiment = SimpleNamespace(optimizer=spec, evaluate=SimpleNamespace(batch=2),
                                             budget=SimpleNamespace(updates=8),
                                             seeds=SimpleNamespace(model=0))
                stepper = EagerStepper(experiment, task, source, None)
                manual = lean_optim.AMSGradMD(
                    reference, lr=base.lr, gain_rate=base.gain_rate,
                    auxiliary_rate=base.auxiliary_rate, direction_rate=base.direction_rate,
                    guard_sigma=.5 if guarded else None, matrix_params=[reference.weight])
                for factor in (0., .25, 1., .5, 0.):
                    before = [p.detach().clone() for p in source.parameters()]
                    for name in ("lr", "gain_rate", "auxiliary_rate", "direction_rate"):
                        peak = getattr(base, name)
                        manual.param_groups[0][name] = None if peak is None else factor * peak
                    manual.zero_grad()
                    task.loss(*task.forward(reference, task.inputs(torch.arange(2)))).backward()
                    manual.step()
                    stepper.step(parts, base.lr * factor, False)
                    for p, q, old in zip(source.parameters(), reference.parameters(), before, strict=True):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        if not factor:
                            self.assertTrue(torch.equal(p, old))
                saved = copy.deepcopy(stepper.state_dict())
                # Older checkpoints had no nominal rate; the destination adapter
                # must supply it so warmup stays effective after loading.
                saved["param_groups"][0].pop("initial_lr")
                stepper.load_state_dict(saved)
                self.assertEqual(stepper.optimizer.param_groups[0]["initial_lr"], base.lr)
                before = [p.detach().clone() for p in source.parameters()]
                stepper.step(parts, 0., False)
                self.assertTrue(all(torch.equal(p, old) for p, old in zip(source.parameters(), before)))

    def test_live_tensor_rates_and_explicit_historical_independent_rates(self):
        source = model()
        rate = torch.tensor(0., dtype=torch.float64)
        optimizer = build_optimizer(AMSGradMD(lr=.125), source, rate=rate)
        before = [p.detach().clone() for p in source.parameters()]
        for p in source.parameters():
            p.grad = torch.full_like(p, .125)
        optimizer.step()
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(source.parameters(), before)))
        self.assertIs(optimizer.param_groups[0]["lr"], rate)
        historical = build_optimizer(AMSGradMD(lr=.125, schedule_components=False), source)
        for p in source.parameters():
            p.grad = torch.full_like(p, .125)
        historical.step()
        self.assertTrue(all(not torch.equal(p, old) for p, old in zip(source.parameters(), before)))
        with self.assertRaisesRegex(ValueError, "boolean"):
            AMSGradMD(lr=.125, schedule_components=1)


if __name__ == "__main__":
    unittest.main()
