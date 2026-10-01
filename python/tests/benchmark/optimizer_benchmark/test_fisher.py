import unittest
import torch

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.fisher import AdaFisherOptimizer, FactorCollector, minmax


class FisherTests(unittest.TestCase):
    def test_actual_fc_measurements_and_first_bias_corrected_update(self):
        layer = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.fill_(1)
        opt = AdaFisherOptimizer(layer, 0.01, damping=0.5)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float64)
        derivatives = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        opt.zero_grad()
        (layer(inputs) * derivatives).sum().backward()
        h, s = opt.collector.measurements[layer.weight]
        torch.testing.assert_close(h, torch.tensor([10.0, 5.0], dtype=h.dtype))
        torch.testing.assert_close(s, torch.tensor([10.0, 20.0], dtype=s.dtype))
        gradient = torch.tensor([[10.0, 5.0], [14.0, 8.0]], dtype=h.dtype)
        torch.testing.assert_close(layer.weight.grad, gradient)
        opt.step()
        metric = torch.tensor([[0.5, 0.5], [1.5, 0.5]], dtype=h.dtype)
        torch.testing.assert_close(layer.weight, 1 - 0.01 * gradient / metric)
        torch.testing.assert_close(opt.state[layer.weight]["h"], 0.8 * h)
        torch.testing.assert_close(opt.state[layer.weight]["s"], 0.8 * s)
        self.assertFalse(opt.state[layer.weight]["h"].requires_grad)
        opt.close()

    def test_constant_factor_and_extreme_range_are_defined(self):
        torch.testing.assert_close(minmax(torch.ones(4)), torch.zeros(4))
        values = torch.tensor([-2.0, 0.0, 6.0])
        torch.testing.assert_close(minmax(values), torch.tensor([0.0, 0.25, 1.0]))

    def test_validation_does_not_overwrite_training_factors(self):
        model = torch.nn.Linear(2, 2, bias=False)
        collector = FactorCollector(model)
        model(torch.ones(2, 2)).sum().backward()
        h, s = (t.clone() for t in collector.measurements[model.weight])
        model.eval()
        with torch.no_grad():
            model(torch.full((2, 2), 100.0))
        torch.testing.assert_close(collector.measurements[model.weight][0], h)
        torch.testing.assert_close(collector.measurements[model.weight][1], s)
        collector.close()
        self.assertEqual(len(model._forward_hooks), 0)

    def test_source_identity_fallback_leaves_damping(self):
        class Temperature(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = torch.nn.Parameter(torch.ones(3))

        model = Temperature()
        opt = AdaFisherOptimizer(model, 0.01, damping=0.1)
        model.temperature.grad = torch.tensor([1.0, 2.0, 3.0])
        opt.step()
        torch.testing.assert_close(model.temperature, torch.tensor([0.9, 0.8, 0.7]))
        opt.close()
