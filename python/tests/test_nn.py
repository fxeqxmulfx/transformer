"""The built models are the historical models, bit for bit.

Golden values: `fixtures/legacy_modular.json`, produced by the historical
code itself (its `source` field names the commit). CPU values hold for this
torch build; CUDA values also need the GPU the fixture names.
"""

import hashlib
import json
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from lab.domain.benchmarks import ModularDivision
from lab.domain.model import Softmax, Sparsemax
from lab.domain.spec import substitute
from lab.infrastructure.benchmarks.modular import make_corpus
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.legacy import import_state, rename

from examples import gptmini, reference

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_modular.json").read_text())
SPECS = {"gptmini": gptmini(width=32, depth=2, heads=4),
         "gptmini-sparsemax": substitute(gptmini(width=32, depth=2, heads=4), Softmax, Sparsemax()),
         "reference": reference(width=32, depth=2, heads=4)}


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def renamed(hashes):
    """Historical names mapped to current ones, in the historical order."""
    return [(rename(name), value) for name, value in hashes.items()]


class LegacyModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def measure(self, name, device):
        expected = FIXTURE["models"][name]
        model = build_model(SPECS[name], expected["vocab"], seed=0)
        parameters = [(key, sha(value)) for key, value in model.named_parameters()]
        model = model.to(device)
        batch = torch.tensor(make_corpus(ModularDivision(prime=11, train_fraction=.2), 0).train, device=device)
        output = model(batch[:, :-1])
        loss = F.cross_entropy(output[:, 4:].reshape(-1, output.shape[-1]), batch[:, 5:].reshape(-1))
        loss.backward()
        gradients = [(key, sha(value.grad)) for key, value in model.named_parameters()]
        return parameters, sha(output), loss.item().hex(), gradients

    def test_initialization_forward_and_gradients_on_cpu(self):
        for name in SPECS:
            with self.subTest(model=name):
                expected = FIXTURE["models"][name]
                parameters, logits, loss, gradients = self.measure(name, "cpu")
                self.assertEqual(parameters, renamed(expected["parameters"]))
                self.assertEqual(logits, expected["logits"])
                self.assertEqual(loss, expected["loss"])
                self.assertEqual(gradients, renamed(expected["gradients"]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_forward_and_gradients_on_the_fixture_gpu(self):
        if FIXTURE["source"].get("gpu") != torch.cuda.get_device_name():
            self.skipTest("CUDA golden values were recorded on another GPU")
        for name in SPECS:
            with self.subTest(model=name):
                expected = FIXTURE["models"][name]["cuda"]
                _, logits, loss, gradients = self.measure(name, "cuda")
                self.assertEqual(logits, expected["logits"])
                self.assertEqual(loss, expected["loss"])
                self.assertEqual(gradients, renamed(expected["gradients"]))

    def test_historical_state_loads_under_current_names(self):
        model = build_model(SPECS["reference"], 149, seed=1)
        state = {"embedding.weight": torch.zeros(149, 32), "position_encoding": torch.zeros(50, 32),
                 "self_attn_mask": torch.ones(50, 50),
                 "decoder.blocks.1.self_attn.attn_heads.3.Wv.weight": torch.zeros(8, 32),
                 "decoder.blocks.0.ffn.ffn.2.weight": torch.zeros(32, 128), "linear.weight": torch.zeros(149, 32)}
        imported = import_state(state)
        self.assertEqual(sorted(imported), ["blocks.0.ffn.output.weight",
                                            "blocks.1.attention.projections.heads.3.value.weight",
                                            "embed.weight", "readout.weight"])
        current = dict(model.named_parameters())
        self.assertTrue(all(current[key].shape == value.shape for key, value in imported.items()))
        with self.assertRaisesRegex(KeyError, "decoder.extra"):
            import_state({"decoder.extra": torch.zeros(1)})


class CorpusTests(unittest.TestCase):
    """Golden split fingerprints of `paper_reproduction.modular_data` at seed 0."""

    def test_the_scientific_splits_are_unchanged(self):
        golden = {(97, .2): (1862, 7450, 239, "492c6e09d5c8d97dde0f9e71344689edf00a19a6aca1f5ef3d7de31ab78376dc",
                             "dfeb50f291c27a7e385d90f5c747da6b0a2347eb305c226d55d7cefa5dd5b0f0"),
                  (193, .25): (9264, 27792, 335, "5a0b7e90d546dfbad53ea44b64e2c3d1fb52af3a9a5640eff40f3d0ba3028ad8",
                               "bba4c2d1ddcc6a23833b1831a59eb6db04371f20dc7054aa145e22d1e602afa4")}
        for (prime, fraction), values in golden.items():
            summary = make_corpus(ModularDivision(prime=prime, train_fraction=fraction), 0).summary()
            self.assertEqual((summary["train_examples"], summary["heldout_examples"], summary["vocab_size"],
                              summary["train_fingerprint"], summary["heldout_fingerprint"]), values)


if __name__ == "__main__":
    unittest.main()
