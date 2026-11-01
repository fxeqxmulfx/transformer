"""Full-state source-transformer forks and the long-phase wall-time cap."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch

from lab.dsl import (Budget, CheckpointHistory, Eager, Evaluate, Experiment, ModularChain, ModularPairs,
                     Schedule, ScheduleFreeAdamW, ScheduleFreeContinuation, Seeds, SourceTransformer, swap)
from test_checkpoint_history import train

ROOT = Path(__file__).resolve().parents[2]


class SourceContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_state_fork_matches_uninterrupted_pairs_and_chain(self):
        for task in (ModularPairs(7, "division"), ModularChain(7, 1, 12, 6, 6)):
            spec = Experiment(model=SourceTransformer(8, 2, 2, 5), benchmark=task,
                              optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1),
                              schedule=Schedule(2), budget=Budget(6, 8), seeds=Seeds(31, 11, 11),
                              evaluate=Evaluate(2, 16), checkpoint=CheckpointHistory(2),
                              execution=Eager(device="cpu"))
            with self.subTest(task=task), tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
                train(spec, source)
                path = Path(source) / "run/checkpoints/step-00000004.pt"
                before = path.read_bytes()
                continued = swap(spec, "checkpoint", ScheduleFreeContinuation(2, str(path),
                                 hashlib.sha256(before).hexdigest()))
                train(continued, target)
                for relative in ("checkpoints/step-00000004.pt", "checkpoint.pt"):
                    got = torch.load(Path(target) / "run" / relative, weights_only=True)
                    want = torch.load(Path(source) / "run" / relative, weights_only=True)
                    for name in ("model", "optimizer", "permutation"):
                        torch.testing.assert_close(got[name], want[name], rtol=0, atol=0)
                    for name in ("step", "cursor", "rotary_random_state", "examples_seen", "last_batch_size"):
                        self.assertEqual(got[name], want[name])
                self.assertEqual(before, path.read_bytes())

    def test_budget_supervisor_stops_only_its_child_and_keeps_output(self):
        spec = importlib.util.spec_from_file_location("bounded_phase",
            ROOT / "experiments/modular_composition_long/run_bounded.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            code, expired = module.run_child([sys.executable, "-c", "print('complete')"],
                                             directory, log, time.monotonic() + 6)
            self.assertEqual((code, expired), (0, False))
            self.assertEqual(log.read_text(), "complete\n")
            started = time.monotonic()
            code, expired = module.run_child([sys.executable, "-c",
                "import time; print('started', flush=True); time.sleep(20)"],
                directory, log, started + 5.5)
            self.assertTrue(expired)
            self.assertLess(code, 0)
            self.assertLess(time.monotonic() - started, 5.5)
            self.assertIn("started", log.read_text())


if __name__ == "__main__":
    unittest.main()
