"""The lab delegates arithmetic and records optimizer code in trajectory identity."""

import hashlib
from pathlib import Path
import unittest

import lean_optim
import torch

from lab.domain.optimizers import AMSGradMD, AMSGradW, Guarded, Inverse, InverseSqrt, Magma, Muon
from lab.infrastructure.optim import build_optimizer
from lab.infrastructure.provenance import code


class PackageIntegrationTests(unittest.TestCase):
    def test_adapter_preserves_hidden_matrix_names_and_scope(self):
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(3, 2)
        model.hidden = torch.nn.Linear(2, 2, bias=False)
        model.hidden_matrices = lambda: [model.hidden.weight]
        optimizer = build_optimizer(Magma(Guarded(Muon(lr=0.1))), model, updates=3, seed=7)
        self.assertIsInstance(optimizer, lean_optim.Muon)
        self.assertEqual(optimizer.hidden, {model.hidden.weight})
        self.assertEqual(optimizer.stages[-1].masked, {model.hidden.weight: "hidden.weight"})
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_adapter_converts_coefficient_values_without_changing_domain_types(self):
        model = torch.nn.Linear(2, 1)
        spec = AMSGradW(lr=0.1, betas=(0.5, 0.75), weight_decay=0.2,
                        beta1_decay=Inverse(), lr_decay=InverseSqrt())
        optimizer = build_optimizer(spec, model)
        self.assertIsInstance(optimizer, lean_optim.AMSGradW)
        self.assertIsInstance(optimizer.spec.beta1_decay, lean_optim.Inverse)
        self.assertIsInstance(spec.beta1_decay, Inverse)
        with self.assertRaises(ValueError):
            Guarded(AMSGradMD(lr=0.1), sigma=0.75)

    def test_external_optimizer_source_is_part_of_checkpoint_engine_identity(self):
        sources = code()
        package = Path(lean_optim.__file__).resolve().parent
        path = package / "coordinate.py"
        self.assertEqual(sources["lean_optim/coordinate.py"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn("lean_optim/optimistic.py", sources)
        self.assertTrue(all(not Path(name).is_absolute() for name in sources))


if __name__ == "__main__":
    unittest.main()
