"""Independent modular arithmetic, answer causality, and exact checkpoint resume."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, logits, make_model, train, transition
from experiments.synthetic_trainers.paper_reproduction.modular_data import answer, complete_rows, fingerprint, make_corpus


class ModularDataTests(unittest.TestCase):
    def test_complete_domain_and_independent_inverse_oracle(self):
        corpus = make_corpus()
        combined = np.concatenate((corpus.train, corpus.heldout))
        self.assertEqual(combined.shape, (97 * 96, 7))
        self.assertEqual(len({tuple(row[:5]) for row in combined}), 97 * 96)
        self.assertFalse({tuple(row[:5]) for row in corpus.train} & {tuple(row[:5]) for row in corpus.heldout})
        for row in combined:
            numerator, denominator, quotient = (int(corpus.tokens[row[index]]) for index in (1, 3, 5))
            self.assertEqual(answer(numerator, denominator, 97), quotient)
        # Computed independently by executing the pinned author's data methods.
        self.assertEqual(fingerprint(combined), "1ad14f76fc72c9125ec1c868bc99f2b4bf760b23beb84d4afc49ef53f470215e")
        self.assertEqual((len(corpus.train), len(corpus.heldout), len(corpus.tokens)), (1862, 7450, 239))

    def test_composite_modulus_and_zero_denominator_are_rejected(self):
        with self.assertRaises(ValueError):
            complete_rows(15)
        with self.assertRaises(ValueError):
            answer(0, 0, 97)
        with self.assertRaises(ValueError):
            make_corpus(2, .01)



class GrokkingTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_answer_token_cannot_leak_to_its_prediction(self):
        corpus = make_corpus(7, .5)
        tokens = torch.tensor(corpus.train[:3, :-1])
        altered = tokens.clone()
        altered[:, 5] = (altered[:, 5] + 1) % len(corpus.tokens)
        for kind in ("reference", "gptmini"):
            with self.subTest(model=kind):
                config = RunConfig(model=kind, prime=7, width=8, heads=1, layers=1, steps=1, device="cpu")
                model = make_model(config, len(corpus.tokens)).eval()
                with torch.no_grad():
                    torch.testing.assert_close(logits(model, tokens)[:, 4], logits(model, altered)[:, 4], rtol=0, atol=0)

    def test_checkpoint_resume_matches_uninterrupted_updates_for_both_optimizers(self):
        config = RunConfig(prime=7, train_fraction=.5, width=8, heads=1, layers=1,
                           steps=6, eval_every=1, batch_size=8, device="cpu", warmup_steps=2)
        for kind, optimizer in (("reference", "adamw"), ("gptmini", "amsgradw")):
            with self.subTest(model=kind, optimizer=optimizer), tempfile.TemporaryDirectory() as root:
                current = replace(config, model=kind, optimizer=optimizer)
                uninterrupted = Path(root) / "full"
                resumed = Path(root) / "resumed"
                train(current, uninterrupted)
                train(replace(current, steps=3), resumed)
                report = train(current, resumed, resume=True)
                full = torch.load(uninterrupted / "checkpoint.pt", weights_only=True)
                resumed_state = torch.load(resumed / "checkpoint.pt", weights_only=True)
                for name, value in full["model"].items():
                    torch.testing.assert_close(value, resumed_state["model"][name], rtol=0, atol=0)
                self.assertEqual(full["examples_seen"], resumed_state["examples_seen"])
                self.assertEqual([point["step"] for point in report["history"]], list(range(7)))
                self.assertEqual(report["plan"]["budget_extensions"], [{"old_steps": 3, "new_steps": 6, "scope": "posthoc_budget_extension"}])
                saved = json.loads((resumed / "measurements.json").read_text())
                self.assertEqual(saved["completed_steps"], 6)
                with self.assertRaises(ValueError):
                    train(replace(current, weight_decay=.5), resumed, resume=True)

    def test_delayed_generalization_requires_a_sustained_streak(self):
        config = RunConfig(steps=5, device="cpu")
        history = [{"step": step, "train": {"accuracy": train_acc}, "heldout": {"accuracy": val_acc}}
                   for step, train_acc, val_acc in ((0, 0, 0), (1, 1, .1), (2, 1, 1), (3, 1, .5), (4, 1, 1), (5, 1, 1))]
        self.assertEqual(transition(history, config)["lag_steps"], 3)
        self.assertEqual(transition(history, config)["heldout_confirmed_step"], 5)
        self.assertIsNone(transition(history[:5], config)["heldout_onset_step"])
