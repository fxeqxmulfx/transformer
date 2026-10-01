"""Entropy baselines, normalized coding, empirical witnesses, and transfer lags."""

from dataclasses import replace
import itertools
import math
import unittest

import torch
from torch import nn

from experiments.synthetic_trainers.compression import code_lengths, membership_auc, prefix_extraction, uniform_compression
from experiments.synthetic_trainers.config import TrainConfig
from experiments.synthetic_trainers.curves import curve_witness, delayed_generalization, fit_value, increase_witness
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.label_noise import noisy_split, noise_fit_metrics
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.records import example_from_targets
from experiments.synthetic_trainers.sequence_oracles import generation_limit, generation_problem_size
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers import vocabulary as v


class UniformControl(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, tokens):
        spec = self.spec
        logits = torch.full((*tokens.shape, spec.vocab_size), -80.0, device=tokens.device)
        logits[:, :, v.IDENTITY_BASE:spec.number_base] = 0
        lengths = tokens[:, 1] - spec.number_base
        ending = torch.arange(tokens.shape[1], device=tokens.device)[None, :] == lengths[:, None] + 2
        logits[ending] = -80
        logits[:, :, v.EOS][ending] = 0
        return logits


class StudyMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uniform_reference_has_zero_gain_and_exact_conditional_entropy(self):
        spec = TaskSpec(task="random_lm", length=5, min_length=2, symbols=4, number_limit=16)
        rows = build_split(spec, "train", 0, 9).examples
        model = UniformControl(spec).train()
        for size in (1, 4, 12):
            scores = code_lengths(model, rows, size)
            report = uniform_compression(rows, scores, spec.symbols, 100)
            self.assertAlmostEqual(report["reference_entropy_bits"], sum(len(row.answer) - 1 for row in rows) * 2)
            self.assertAlmostEqual(report["net_gain_bits"], 0, places=5)
            self.assertAlmostEqual(report["mixture_gain_bits"], 0, places=5)
            self.assertAlmostEqual(report["eos_code_bits"], 0, places=5)
            self.assertTrue(model.training)
        for row in rows:
            self.assertEqual(generation_limit(row.prompt, spec), len(row.answer))
            self.assertEqual(generation_problem_size(row.prompt, spec), len(row.answer) - 1)

    def test_clipping_happens_per_sequence_and_mixture_has_at_most_one_bit_overhead(self):
        spec = TaskSpec(task="random_lm", length=4, symbols=4)
        rows = build_split(spec, "train", 0, 2).examples
        scores = [{"bits": 4.0, "payload_bits": 3.0}, {"bits": 20.0, "payload_bits": 19.0}]
        report = uniform_compression(rows, scores, 4, 10)
        self.assertEqual(report["net_gain_bits"], -8)
        self.assertEqual(report["clipped_sequence_gain_bits"], 4)
        self.assertGreaterEqual(report["mixture_code_bits"], 4 + 8)
        self.assertLessEqual(report["mixture_code_bits"], 4 + 8 + 2)
        self.assertAlmostEqual(report["net_bits_per_parameter"], -0.8)
        scores[1] = {"bits": 1e9, "payload_bits": 1e9}
        self.assertTrue(math.isfinite(uniform_compression(rows, scores, 4, 10)["mixture_code_bits"]))

    def test_uniform_capacity_reference_cannot_be_applied_to_algorithmic_answers(self):
        rows = build_split(TaskSpec(task="copy", length=4), "train", 0, 2).examples
        with self.assertRaises(ValueError):
            uniform_compression(rows, [{"bits": 0}] * 2, 4, 10)

    def test_membership_auc_ties_and_direction(self):
        def scores(values):
            return [{"bits_per_target": value} for value in values]
        self.assertEqual(membership_auc(scores([1, 1]), scores([1, 2])), 0.75)
        self.assertEqual(membership_auc(scores([0, 1]), scores([2, 3])), 1.0)
        self.assertEqual(membership_auc(scores([2, 3]), scores([0, 1])), 0.0)
        self.assertEqual(membership_auc(scores([1]), scores([1])), 0.5)
        with self.assertRaises(ValueError):
            membership_auc([], scores([1]))

    def test_per_example_loss_is_not_token_weighted_loss(self):
        class Fixed(nn.Module):
            def forward(self, tokens):
                return torch.tensor([[0.0, 0.0], [math.log(3), 0.0]])[tokens]
        rows = [example_from_targets("lookup", (0, 0), (0, 0)), example_from_targets("lookup", (1,), (0,))]
        for size in (1, 3):
            report = evaluate(Fixed(), rows, size)
            self.assertAlmostEqual(report["example_loss"], (math.log(2) + math.log(4 / 3)) / 2, places=6)
            self.assertAlmostEqual(report["loss"], (2 * math.log(2) + math.log(4 / 3)) / 3, places=6)

    def test_noise_fit_distinguishes_memorized_noise_from_clean_labels(self):
        spec = TaskSpec(task="copy", length=4, symbols=4)
        clean = build_split(spec, "train", 0, 3)
        observed, info = noisy_split(clean, 1, 4)
        table = {row.tokens[:i + 1]: target for row in observed.examples for i, target in enumerate(row.targets) if target != v.IGNORE}
        class Memorizer(nn.Module):
            def forward(self, tokens):
                logits = torch.full((*tokens.shape, spec.vocab_size), -20.0)
                for r, stream in enumerate(tokens.tolist()):
                    for i in range(len(stream)):
                        logits[r, i, table.get(tuple(stream[:i + 1]), v.EOS)] = 20.0
                return logits
        model = Memorizer().train()
        report = noise_fit_metrics(model, observed.examples, clean.examples, 2)
        self.assertEqual(report["corrupted_targets"], info["changed_targets"])
        self.assertEqual(report["observed_label_accuracy"], 1)
        self.assertEqual(report["clean_label_accuracy"], 0)
        self.assertTrue(model.training)

    def test_partial_prefix_extraction_leaves_payload_to_predict_and_restores_mode(self):
        spec = TaskSpec(task="random_lm", length=4, symbols=2)
        rows = build_split(spec, "train", 3, 5).examples
        model = UniformControl(spec).train()
        results = prefix_extraction(model, rows, 2, fractions=(0, 0.5, 0.999))
        self.assertEqual([item["provided_payload_tokens"] for item in results], [0, 10, 15])
        self.assertEqual([item["suffix_target_tokens"] for item in results], [25, 15, 10])
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            prefix_extraction(model, rows, 2, fractions=(1,))

    def test_curve_witness_matches_an_independent_exhaustive_search(self):
        for ys in itertools.product(range(3), repeat=5):
            expected = any(ys[b] < ys[a] and ys[b] < ys[c] and ys[d] < ys[c]
                           for a, b, c, d in itertools.combinations(range(5), 4))
            result = curve_witness(list(enumerate(ys)))
            self.assertEqual(result is not None, expected)
        strong = curve_witness(list(enumerate((3, 1, 4, 0))))
        self.assertTrue(strong["corrects_first_minimum"])
        self.assertIsNone(curve_witness(list(enumerate((3, 1, 1.001, 0))), 0.01))
        self.assertIsNotNone(increase_witness([(2, .1), (4, .3)], 0.01))
        self.assertIsNone(increase_witness([(2, .3), (4, .1)], 0.01))
        self.assertIsNone(curve_witness([(i, -i) for i in range(10_000)]))
        for points in ([(0, 1), (0, 2)], [(1, 2), (0, 1)], [(0, float("nan"))]):
            with self.assertRaises(ValueError):
                curve_witness(points)

    def history(self):
        def scores(accuracy):
            return {"sequence_accuracy": accuracy, "token_accuracy": accuracy, "example_loss": 1 - accuracy}
        rows = []
        for step, clean, observed, generalization in zip((0, 5, 10, 20, 30, 40), (0, 1, 1, 1, 1, 1),
                                                        (0, 0.5, 1, 1, 1, 1), (0, 0.1, 1, 0.3, 1, 1)):
            rows.append({"step": step, "training_seconds": step / 2, "epochs_seen": step / 10,
                         "train": scores(observed), "train_clean": scores(clean),
                         "validation": scores(1), "validation_ood": {"length-8": scores(generalization)}})
        return rows

    def test_transfer_lag_requires_stable_ood_success_and_records_both_train_fits(self):
        report = delayed_generalization(self.history(), TrainConfig(study="memorization", target=0.9))
        self.assertEqual(report["clean_train_fit_step"], 5)
        self.assertEqual(report["observed_train_fit_step"], 10)
        self.assertEqual(report["generalization_step"], 30)
        self.assertEqual(report["confirmed_at_step"], 40)
        self.assertEqual(report["lag_steps"], 25)
        self.assertEqual(report["lag_epochs"], 2.5)
        self.assertTrue(report["delayed_transfer_candidate"])

    def test_id_generalization_is_separate_from_length_transfer(self):
        history = self.history()
        for row in history:
            row["validation_novel"] = {**row["validation"], "sequence_accuracy": float(row["step"] >= 20)}
            row["validation_ood"] = {"length-8": {**row["validation"], "sequence_accuracy": 0.5}}
        report = delayed_generalization(history, TrainConfig(study="memorization", target=0.9))
        self.assertEqual(report["id_generalization_step"], 20)
        self.assertEqual(report["id_confirmed_at_step"], 30)
        self.assertEqual(report["id_lag_steps"], 15)
        self.assertTrue(report["delayed_id_generalization_candidate"])
        self.assertIsNone(report["generalization_step"])
        self.assertFalse(report["delayed_transfer_candidate"])

    def test_id_generalization_requires_novel_inputs_and_excludes_random_control(self):
        history = self.history()
        config = TrainConfig(study="memorization", target=0.9)
        self.assertIsNone(delayed_generalization(history, config, random_control=True)["id_generalization_step"])
        for row in history:
            row["validation_novel"] = None
        self.assertIsNone(delayed_generalization(history, config)["id_generalization_step"])

    def test_absent_ood_or_incomplete_budget_cannot_confirm_transition(self):
        config = TrainConfig(study="memorization", target=0.9)
        self.assertIsNone(delayed_generalization(self.history()[:-1], config)["generalization_step"])
        history = [{**row, "validation_ood": {}} for row in self.history()]
        self.assertIsNone(delayed_generalization(history, config)["generalization_step"])
        self.assertIsNone(delayed_generalization(self.history(), config, random_control=True)["generalization_step"])
        self.assertIsNone(delayed_generalization(self.history(), replace(config, target=None))["generalization_step"])

    def test_interpolation_metric_uses_observed_teacher_forcing(self):
        metrics = {"sequence_accuracy": 0, "teacher_forced": {"sequence_accuracy": 1, "token_accuracy": 0.9, "example_loss": 0.2}}
        self.assertEqual(fit_value(metrics, "example_error"), 0)
        self.assertAlmostEqual(fit_value(metrics, "token_error"), 0.1)
        self.assertEqual(fit_value(metrics, "loss"), 0.2)

    def test_repeated_validation_inputs_or_empty_novel_probes_cannot_confirm_transfer(self):
        config = TrainConfig(study="memorization", target=0.9)
        history = self.history()
        for row in history:
            row["validation_novel"] = None
        self.assertIsNone(delayed_generalization(history, config)["generalization_step"])
        history = self.history()
        for row in history:
            row["validation_ood_novel"] = {"length-8": row["validation_ood"]["length-8"], "position_shift": None}
        self.assertIsNone(delayed_generalization(history, config)["generalization_step"])
