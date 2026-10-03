"""Explicit softmax/sparsemax factories for the native complementary AdamW path.

Only attention normalization changes. Serialization, sampling, all-parameter
decay, warmup, the selected fixed rate schedule, validation selection and separate
novel-ID/length scores remain owned by the existing complementary trainer.
This fresh-run adapter does not select a scientific architecture or open a gate.
"""

from pathlib import Path

from experiments.gpt_mini import GPTMini
from . import complementary_training
from .complementary_attention_report import description
from .runtime import write_json
from .sparsemax_attention import replace_attention


def softmax_model(config):
    return GPTMini(config)


def sparsemax_model(config):
    return replace_attention(GPTMini(config))


def train(config, spec, model_spec, directory, *, attention_normalization, progress=None, prepared_splits=None):
    """Require an explicit normalizer before any model construction or writes.

    Both factories consume exactly the original constructor's random draws and
    preserve its parameter shapes and ties. The common loop then applies its
    unchanged explicit matrix initialization. Interrupted fresh runs are kept;
    this adapter never restarts or overwrites an incomplete run.
    """
    tag = description(attention_normalization)
    factory = {"softmax": softmax_model, "sparsemax": sparsemax_model}[attention_normalization]
    result = complementary_training.train(config, spec, model_spec, directory,
        model_factory=factory, progress=progress, prepared_splits=prepared_splits)
    if result["provenance"]["factory"] != tag["factory"]:
        raise ValueError("The complementary normalizer factory differs from the actual run")
    result["complementary_attention"] = tag
    write_json(Path(directory) / "result.json", result)
    return result
