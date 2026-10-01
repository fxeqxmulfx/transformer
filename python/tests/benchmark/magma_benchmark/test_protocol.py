"""Integration and provenance contracts before the benchmark is launched."""

import copy
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import training_starts
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import state_hash
from gpt_mini.infrastructure.benchmark.magma_benchmark.protocol import baseline, check_pairing, check_compatible
from gpt_mini.infrastructure.benchmark.magma_benchmark.registry import METHODS, make_optimizer


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_baseline_and_equal_new_tuning_budgets(self):
        rows, metadata, provenance = baseline()
        self.assertEqual(len(rows), 216)
        self.assertEqual(len([r for r in rows if r["phase"] == "final"]), 108)
        self.assertEqual(len(provenance["raw_sha256"]), 64)
        self.assertEqual(len(METHODS), 6)
        self.assertTrue(all(len(set(m.rates)) == 3 and min(m.rates) > 0 for m in METHODS))
        check_compatible(metadata["protocol"], metadata["protocol"])

    def test_incompatible_configuration_is_rejected(self):
        _, metadata, _ = baseline()
        changed = copy.deepcopy(metadata["protocol"])
        changed["batch"] = 16
        with self.assertRaisesRegex(ValueError, "batch"):
            check_compatible(changed, metadata["protocol"])

    def test_unpaired_batches_are_rejected_before_combining_rankings(self):
        rows, _, _ = baseline()
        reference = next(r for r in rows if r["phase"] == "final")
        new = dict(reference, method="magma_adamw")
        check_pairing([new], rows)
        new["batch_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "batch_sha256"):
            check_pairing([new], rows)

    def test_same_actual_model_and_batch_plan_as_old_baseline(self):
        rows, metadata, _ = baseline()
        cfg = Config(**metadata["protocol"]["config"])
        for seed in range(3):
            model = make_model(cfg, "softmax", seed)
            old = next(r for r in rows if r["phase"] == "final" and r["seed"] == seed)
            self.assertEqual(state_hash(model), old["initial_sha256"])
            import hashlib
            starts = training_starts(metadata["protocol"]["data_boundaries"][0], 64, 32, 1000, seed)
            self.assertEqual(hashlib.sha256(starts.numpy().tobytes()).hexdigest(), old["batch_sha256"])

    def exercise(self, device):
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        for attention in ("softmax", "sparsemax"):
            for method in METHODS:
                with self.subTest(device=device, attention=attention, method=method.name):
                    model = make_model(cfg, attention, 0, device)
                    model.magma_seed = 0
                    optimizer = make_optimizer(method.name, model, min(method.rates))
                    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=device)
                    targets = tokens.roll(-1, 1)
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())
                        self.assertTrue(torch.isfinite(loss).item())
                        loss.backward()
                        optimizer.step()
                        self.assertTrue(all(p.isfinite().all().item() for p in model.parameters()))
                    self.assertIs(model.embed.weight, model.unembed.weight)
                    if method.name.startswith("magma_"):
                        diag = optimizer.diagnostics()
                        self.assertEqual(diag["mask_draws"], 12)
                        self.assertEqual(len(diag["blocks"]), 4)
                        self.assertTrue(all(p.grad is not None for p in model.parameters()))
                        self.assertEqual(optimizer.base.steps, 3)
                    optimizer.close()

    def test_all_new_optimizers_on_both_attentions_cpu(self):
        self.exercise("cpu")

    @unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is an explicit phase")
    def test_all_new_optimizers_on_both_attentions_cuda(self):
        self.assertTrue(torch.cuda.is_available())
        self.exercise("cuda")
