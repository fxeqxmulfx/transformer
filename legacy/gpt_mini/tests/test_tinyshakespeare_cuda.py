"""Real-data application smoke tests after the numerical contract suite."""

from dataclasses import replace
import os
import tempfile
import unittest

from gpt_mini.application.benchmark import RunBenchmark
from gpt_mini.domain.benchmark import ModelConfig, Request
from gpt_mini.domain.stopping import StopConfig
from gpt_mini.infrastructure.results import FilesystemResults


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class TinyShakespeareCudaTests(unittest.TestCase):
    def test_both_backend_families_train_restore_and_validate_real_data(self):
        from gpt_mini.infrastructure.training import CudaTraining

        class CheckedTraining(CudaTraining):
            def preflight(self):
                # The enclosing suite has already checked all numerical rules.
                self.require_cuda()
                return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

        request = Request(methods=("adamw", "amsgradw"), seeds=(0,), batch=128,
                          model=ModelConfig(n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8),
                          stopping=StopConfig(every=1, min_steps=0, max_steps=2))
        with tempfile.TemporaryDirectory() as directory:
            training = CheckedTraining(progress=lambda _: None)
            result = RunBenchmark(training, FilesystemResults(directory)).execute(request)
            self.assertEqual((result["runs"], result["failed"]), (4, 0))
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["checkpoint_checks"]), 4)
            resumed = RunBenchmark(CheckedTraining(), FilesystemResults(directory)).execute(replace(request, validate_only=True))
            self.assertEqual(resumed["runs"], 4)
