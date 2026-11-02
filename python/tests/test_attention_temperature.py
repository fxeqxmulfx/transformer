"""An isolated score gain preserves initialization and has the declared derivative."""

from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from lab.dsl import CanonicalResidual, ResidualMaxAutotune, TemperatureResidual, swap
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.centered_xsa import centered_projection
from lab.infrastructure.nn.source_transformer import rotate
from lab.infrastructure.loader import load
from test_residual_basis import tiny
from test_residual_transformer import train


class AttentionTemperatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_only_head_gains_are_added_without_changing_initial_values_or_random_stream(self):
        base = CanonicalResidual(8, 2, 2, 8)
        candidate = TemperatureResidual(**vars(base))
        reference = build_model(base, 11, 17)
        reference_rng = torch.get_rng_state().clone()
        actual = build_model(candidate, 11, 17)
        self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
        original = dict(reference.named_parameters())
        extra = {k:v for k,v in actual.named_parameters() if k not in original}
        self.assertEqual(set(extra), {f"blocks.{i}.sa.log_gain" for i in range(2)})
        self.assertEqual(sum(v.numel() for v in extra.values()), 4)
        for name, value in actual.named_parameters():
            expected = original[name] if name in original else torch.zeros_like(value)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertNotIn("mixing", vars(candidate))
        tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 1, 7, 8]])
        reference, actual = reference.double(), actual.double()
        a, b = reference(tokens), actual(tokens)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        weights = torch.linspace(-1, 1, a.numel(), dtype=torch.float64).view_as(a)
        (a * weights).sum().backward()
        (b * weights).sum().backward()
        original = dict(reference.named_parameters())
        for name, value in actual.named_parameters():
            self.assertTrue(torch.isfinite(value.grad).all().item())
            if name in original:
                torch.testing.assert_close(value.grad, original[name].grad, rtol=0, atol=0)
            else:
                self.assertTrue((value.grad.abs() > 0).all().item())

    def test_temperature_matches_explicit_causal_scores_and_finite_difference(self):
        model = build_model(TemperatureResidual(8, 1, 2, 8), 11, 9).double()
        attention = model.blocks[0].sa
        with torch.no_grad():
            attention.log_gain.copy_(torch.tensor([-.4, .6], dtype=torch.float64))
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        cos, sin = model.rope_cos[:5], model.rope_sin[:5]
        q, k, v = attention.c_attn(x).split(8, dim=2)
        q, k, v = (z.view(2, 5, 2, 4).transpose(1, 2) for z in (q, k, v))
        q, k = rotate(attention.q_norm(q), cos, sin), rotate(attention.k_norm(k), cos, sin)
        logits = (q @ k.transpose(-1, -2)) * attention.log_gain.exp().view(1, 2, 1, 1) / 2
        causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
        probabilities = logits.masked_fill(causal, -torch.inf).softmax(-1)
        anchor = v[:, :, :1]
        y = centered_projection(probabilities @ (v - anchor), v - anchor, v)
        y = torch.cat((torch.zeros_like(y[:, :, :1]), y[:, :, 1:]), dim=2)
        expected = attention.c_proj(y.transpose(1, 2).contiguous().view(2, 5, 8))
        actual, valid = attention(x, cos, sin)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        self.assertTrue(valid.all().item())
        gain = attention.log_gain.detach().clone().requires_grad_()
        function = lambda value: torch.func.functional_call(attention, {"log_gain": value}, (x, cos, sin))[0]
        self.assertTrue(torch.autograd.gradcheck(function, (gain,), eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_repeated_prefix_remains_finite_with_nonunit_temperatures(self):
        model = build_model(TemperatureResidual(8, 2, 2, 8), 11, 23).double()
        with torch.no_grad():
            for block in model.blocks:
                block.sa.log_gain.copy_(torch.tensor([-2., 2.]))
        output = model(torch.ones(2, 6, dtype=torch.long))
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(output).all().item())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_compiled_cpu_temperature_state_and_sampler_resume_exactly(self):
        spec = swap(tiny(kind=TemperatureResidual), "execution", ResidualMaxAutotune(device="cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, straight = train(spec, root / "straight")
            train(swap(spec, "budget.updates", 2), root / "continued")
            result, continued = train(spec, root / "continued")
            left, right = straight.checkpoint("cpu"), continued.checkpoint("cpu")
            clocks = {"training_seconds", "diagnostic_seconds", "wall_seconds"}
            torch.testing.assert_close({k:v for k,v in left.items() if k not in clocks},
                                       {k:v for k,v in right.items() if k not in clocks}, rtol=0, atol=0)
            self.assertEqual(straight.records("gradients"), continued.records("gradients"))
            self.assertEqual(result["stop"], {"step": 6, "reason": "budget"})
            self.assertTrue(all("test" not in row for row in continued.records("history")))

    def test_only_model_changes_in_matched_recipes_and_report_is_sealed(self):
        root = Path(__file__).resolve().parents[2]
        folder = root / "experiments/attention_temperature"
        study = load(folder)
        reference = load(root / "experiments/basis_residual")
        for seed in (0, 1):
            run = study.experiments[f"temperature-hard-recall-s{seed}"]
            old = reference.experiments[f"canonical-large-recall-s{seed}"]
            self.assertEqual(swap(run, "model", old.model), old)

        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            result = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result)
            return result

        protocol = module("temperature_protocol", folder / "protocol.py")
        with patch.dict(sys.modules, {"protocol": protocol}):
            reporter = module("temperature_report", folder / "report.py")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "runs").mkdir()
            for status in ("learning", "interrupted", "awaiting_continuation"):
                (temporary / "runs/phase.json").write_text(json.dumps({"status": status, "cases": {}}))
                with patch.object(reporter, "HERE", temporary), \
                     patch.object(reporter, "frozen", return_value={"cases": [{"label": "case"}]}), \
                     self.assertRaisesRegex(ValueError, "sealed"):
                    reporter.report()


if __name__ == "__main__":
    unittest.main()
