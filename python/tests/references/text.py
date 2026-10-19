"""Replay the frozen, uncompiled text trainer and optimizer zoo on today's backend."""

from functools import partial
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from common import hashes


class CpuEvent:
    """The frozen trainer's CUDA timing hook, unused by numerical comparisons."""

    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.0


class CpuCudaTiming:
    """Delegate CUDA queries, replacing only timing calls in the CPU trainer."""

    Event = CpuEvent

    def synchronize(self, *args, **kwargs):
        pass

    def max_memory_reserved(self, *args, **kwargs):
        return 0

    def __getattr__(self, name):
        return getattr(torch.cuda, name)


class CpuTorchTiming:
    cuda = CpuCudaTiming()

    def __getattr__(self, name):
        return getattr(torch, name)


def replay(golden, corpus):
    from gpt_mini import Config
    from gpt_mini.domain.stopping import StopConfig
    from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData
    from gpt_mini.infrastructure.benchmark.full_compile_benchmark import training
    from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark import training as extensions

    data = TextData.load(corpus)
    for name, expected in golden["runs"].items():
        config = expected["config"]
        module = extensions if config.get("family") == "extensions" else training
        step_name = "ExtensionStep" if module is extensions else "FullStep"
        step = partial(getattr(module, step_name), compiled=False)
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(module, step_name, step),
              patch.object(module, "torch", CpuTorchTiming())):
            result = module.train_one(Config(**config["model"]), data, config["attention"],
                                      config.get("method", "sgd"), config["rate"], config["seed"],
                                      config["batch"], StopConfig(**config["stop"]), directory, device="cpu")
            if result["status"] not in ("ok", "recovered") or result["stop_reason"] == "runtime_error":
                raise AssertionError(f"Historical {name} reference failed: {result}")
            for key in expected.keys() & result.keys():
                if key != "config":
                    expected[key] = result[key]
            expected["curves"] = [{key: value for key, value in row.items() if key != "train_seconds"}
                                  for row in result["curves"]]
            state = torch.load(Path(result["checkpoint"]), weights_only=True)
            expected["best"] = hashes(state["model"])
