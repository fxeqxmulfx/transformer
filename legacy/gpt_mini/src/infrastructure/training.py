"""CUDA execution adapter reusing the tested complete-step numerical kernels."""

from dataclasses import asdict
import hashlib
import importlib
import io
import os
import platform
import unittest

import torch

from .gpt_mini import Config
from .benchmark.amsgrad_extensions_benchmark.protocol import METHODS as EXTENSIONS
from .benchmark.amsgrad_extensions_benchmark.training import train_one as train_extension
from .benchmark.amsgrad_extensions_benchmark.validation import check_checkpoint as check_extension, replay_stopping
from .benchmark.full_compile_benchmark.training import train_one as train_original
from .benchmark.full_compile_benchmark.validation import check_checkpoint as check_original
from .benchmark.optimizer_benchmark.data import TextData, training_starts
from .benchmark.paths import DATA_FILE, artifact_path
from .benchmark.suites import test_suite


class CudaTraining:
    def __init__(self, progress=print):
        self.progress = progress
        self.data = self.config = self.stopping = None

    @staticmethod
    def require_cuda():
        if not torch.cuda.is_available():
            raise RuntimeError("The optimizer comparison requires CUDA")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def preflight(self):
        self.require_cuda()
        os.environ["OPTIMIZER_BENCH_CUDA"] = "1"
        suite = test_suite()
        loader = unittest.TestLoader()
        suite.addTests(loader.loadTestsFromModule(importlib.import_module("gpt_mini.infrastructure.benchmark.tests.test_parameter_groups")))
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError("CPU/CUDA contracts failed:\n" + stream.getvalue())
        return {"tests": result.testsRun, "failures": 0, "errors": 0, "skipped": 0, "log": stream.getvalue()}

    def dataset_info(self):
        data = TextData.load(DATA_FILE)
        return {"sha256": data.sha256, "boundaries": list(data.boundaries), "characters": data.characters}

    def environment(self):
        self.require_cuda()
        device = torch.cuda.get_device_properties(0)
        return {"gpu": device.name, "gpu_total_bytes": device.total_memory,
                "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}

    def configure(self, model, batch, stopping):
        self.require_cuda()
        self.config, self.batch, self.stopping = Config(**asdict(model)), batch, stopping
        self.data = TextData.load(DATA_FILE, "cuda")
        if self.config.vocab_size != len(self.data.characters):
            raise ValueError("Vocabulary must match the dataset")

    def run(self, job, artifacts):
        train = train_extension if job.method in EXTENSIONS else train_original
        return train(self.config, self.data, job.attention, job.method, job.rate, job.seed,
                     self.batch, self.stopping, artifacts, progress=self.progress)

    def validate(self, row):
        replay_stopping(row, self.stopping)
        starts = training_starts(len(self.data.train), self.config.max_seq_len, self.batch,
                                 self.stopping.max_steps, row["seed"])
        checks = ((starts, "batch_plan_sha256"), (starts[:row["actual_steps"]], "batch_sha256"))
        for values, field in checks:
            if hashlib.sha256(values.numpy().tobytes()).hexdigest() != row[field]:
                raise ValueError("Saved minibatch stream differs from the paired plan")
        path = artifact_path(row["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["checkpoint_sha256"] or row["test_evaluations"] != 1:
            raise ValueError("Changed checkpoint or repeated test evaluation")
        check = check_extension if row["method"] in EXTENSIONS else check_original
        return check(row, self.config, self.data, self.batch)
