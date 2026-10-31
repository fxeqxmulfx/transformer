"""Checkpoint integrity, unchanged forwards and independent alias-gradient controls."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from lab.dsl import PeriRotate90, PeriRotate90Restart, ScheduleFreeAdamW
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.peri_restart import alias_projection
from lab.infrastructure.optim import build_optimizer


class PeriRestartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "checkpoint.pt"
        self.source = build_model(PeriRotate90(16, 1), 7, 11)
        torch.save({"model": self.source.state_dict(),
                    "optimizer": {"param_groups": [{"training": True}]}}, self.path)
        self.spec = PeriRotate90Restart(PeriRotate90(16, 1), str(self.path),
            hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.inputs = torch.tensor([(a, b) for a in range(7) for b in range(1, 7)])

    def test_exact_query_restore_and_selected_trainable_parameters(self):
        self.spec.check()
        model = build_model(replace(self.spec, train_embedding=False), 7, 23)
        torch.testing.assert_close(model(self.inputs), self.source(self.inputs), rtol=0, atol=0)
        self.assertFalse(model.E.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.hidden_matrices()))
        optimizer = build_optimizer(ScheduleFreeAdamW(lr=.003), model)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 2)
        frozen = model.E.detach().clone()
        optimizer.param_groups[0]["lr"] = .003
        model(self.inputs).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(model.E, frozen, rtol=0, atol=0)

    def test_hash_and_query_mode_are_checked_before_use(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_model(replace(self.spec, sha256="0" * 64), 7, 11)
        state = torch.load(self.path, weights_only=False)
        state["optimizer"]["param_groups"][0]["training"] = False
        torch.save(state, self.path)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "training-query"):
            build_model(replace(self.spec, sha256=digest), 7, 11)

    def test_noise_is_seeded_and_zero_noise_preserves_weights(self):
        a = build_model(replace(self.spec, noise=.03), 7, 17)
        b = build_model(replace(self.spec, noise=.03), 7, 17)
        c = build_model(replace(self.spec, noise=.03), 7, 23)
        for x, y in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(torch.equal(a.E, c.E))
        self.assertFalse(torch.equal(a.E, self.source.E))

    def test_projected_embedding_has_exact_orbits_and_equal_alias_logits(self):
        model = build_model(replace(self.spec, alias_size=3), 7, 11)
        permutation = torch.tensor([2 * q % 7 for q in range(7)])
        torch.testing.assert_close(model.E, model.E[permutation], rtol=0, atol=0)
        torch.testing.assert_close(model.E[0], self.source.E[0], rtol=0, atol=0)
        logits = model(self.inputs)
        # GEMM columns can differ by a few FP32 ulps despite identical rows of E.
        torch.testing.assert_close(logits, logits[:, permutation], rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(alias_projection(model.E, 3), model.E, rtol=1e-6, atol=2e-7)
        double_logits = model.double()(self.inputs)
        torch.testing.assert_close(double_logits, double_logits[:, permutation], rtol=1e-13, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "divide"):
            alias_projection(model.E, 4)

    def test_full_table_gradient_symmetry_and_noncoprime_control(self):
        model = build_model(replace(self.spec, alias_size=3), 7, 11).double()
        for exponent in (2, 3):
            labels = torch.tensor([pow(a * pow(b, -1, 7) % 7, exponent, 7)
                                   for a, b in self.inputs.tolist()])
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(self.inputs), labels).backward()
            gradient = model.E.grad
            transverse = gradient - alias_projection(gradient, 3)
            ratio = float(transverse.norm() / gradient.norm())
            if exponent == 2:
                self.assertLess(ratio, 1e-12)
            else:
                self.assertGreater(ratio, 1e-4)

    def test_invalid_frozen_or_noise_spec(self):
        for changes in ({"train_embedding": False, "train_branch": False},
                        {"noise": -1}, {"noise": float("nan")}, {"sha256": ""}, {"alias_size": 0}):
            with self.assertRaises(ValueError):
                replace(self.spec, **changes)


if __name__ == "__main__":
    unittest.main()
