"""What a memorization study measures of a model and of its splits, on test doubles with exact logits.

The cases of `experiments/synthetic_trainers/tests/test_study_metrics.py` and
`test_study_data.py`.
"""

from dataclasses import replace
import math
import unittest

import torch
from torch import nn

from lab.domain.generative import Copy, RandomLM
from lab.domain.tasks import Problem
from lab.infrastructure.benchmarks.synthetic import generator
from lab.infrastructure.benchmarks.synthetic.compression import (code_lengths, membership_auc, prefix_extraction,
                                                                 uniform_compression)
from lab.infrastructure.benchmarks.synthetic.corpus import context_report, corpus_report
from lab.infrastructure.benchmarks.synthetic.noise import noise_fit, noisy_split
from lab.infrastructure.benchmarks.synthetic.records import example_from_targets
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.splits import build_split, pools
from lab.infrastructure.benchmarks.synthetic.vocabulary import ACCEPT, BOS, EOS, IDENTITY_BASE, IGNORE, REJECT


class UniformControl(nn.Module):
    """Uniform over the random control's symbols, then certain of EOS where the stated length ends."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, self.source.vocab), -80.0)
        logits[:, :, IDENTITY_BASE:self.source.number_base] = 0
        ending = torch.arange(tokens.shape[1])[None, :] == (tokens[:, 1] - self.source.number_base)[:, None] + 2
        logits[ending] = -80
        logits[:, :, EOS][ending] = 0
        return logits


class Memorizer(nn.Module):
    """Certain of the label each supervised context of `examples` has, and of EOS elsewhere."""

    def __init__(self, examples, vocab):
        super().__init__()
        self.vocab = vocab
        self.table = {example.tokens[:position + 1]: target for example in examples
                      for position, target in enumerate(example.targets) if target != IGNORE}

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, self.vocab), -20.0)
        for row, stream in enumerate(tokens.tolist()):
            for position in range(len(stream)):
                logits[row, position, self.table.get(tuple(stream[:position + 1]), EOS)] = 20.0
        return logits


class NotFinite(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, tokens):
        return torch.full((*tokens.shape, self.vocab), math.nan)


def greedy(table, context, limit):
    """The tokens a model certain of `table` generates from `context`, EOS included, at most `limit`."""
    generated = ()
    while len(generated) < limit and EOS not in generated:
        generated += (table.get(tuple(context) + generated, EOS),)
    return generated


def teacher_forced(examples):
    return Rows(examples, "cpu", generate=False)


class CompressionTests(unittest.TestCase):
    def test_the_uniform_code_gains_nothing_and_costs_its_entropy(self):
        source = generator(Problem(RandomLM(symbols=4, number_limit=16), 5, 2))
        examples = build_split(source, "train", 0, 9).examples
        for batch in (1, 4, 12):
            with self.subTest(batch=batch):
                found = uniform_compression(examples, code_lengths(UniformControl(source), teacher_forced(examples),
                                                                   batch), 4, 100)
                self.assertAlmostEqual(found["reference_entropy_bits"],
                                       sum(len(example.answer) - 1 for example in examples) * 2)
                for key in ("net_gain_bits", "mixture_gain_bits", "eos_code_bits"):
                    self.assertAlmostEqual(found[key], 0, places=5)

    def test_clipping_is_per_sequence_and_the_mixture_costs_at_most_a_bit_more(self):
        examples = build_split(generator(Problem(RandomLM(symbols=4), 4, None)), "train", 0, 2).examples
        scores = [{"bits": 4.0, "payload_bits": 3.0}, {"bits": 20.0, "payload_bits": 19.0}]
        found = uniform_compression(examples, scores, 4, 10)
        self.assertEqual((found["net_gain_bits"], found["clipped_sequence_gain_bits"]), (-8, 4))
        self.assertTrue(4 + 8 <= found["mixture_code_bits"] <= 4 + 8 + 2)
        self.assertAlmostEqual(found["net_bits_per_parameter"], -.8)
        scores[1] = {"bits": 1e9, "payload_bits": 1e9}
        self.assertTrue(math.isfinite(uniform_compression(examples, scores, 4, 10)["mixture_code_bits"]))

    def test_a_code_length_that_is_not_finite_is_none(self):
        source = generator(Problem(RandomLM(symbols=4), 4, None))
        examples = build_split(source, "train", 0, 2).examples
        self.assertIsNone(code_lengths(NotFinite(source.vocab), teacher_forced(examples), 2))

    def test_membership_ranks_members_below_nonmembers_and_splits_ties(self):
        def scores(values):
            return [{"bits_per_target": value} for value in values]

        self.assertEqual(membership_auc(scores([1, 1]), scores([1, 2])), .75)
        self.assertEqual(membership_auc(scores([0, 1]), scores([2, 3])), 1)
        self.assertEqual(membership_auc(scores([2, 3]), scores([0, 1])), 0)
        self.assertEqual(membership_auc(scores([1]), scores([1])), .5)

    def test_a_partial_prefix_leaves_payload_to_generate(self):
        source = generator(Problem(RandomLM(symbols=2), 4, None))
        examples = build_split(source, "train", 3, 5).examples
        found = prefix_extraction(UniformControl(source), examples, 2, "cpu", fractions=(0, .5, .999))
        self.assertEqual([item["provided_payload_tokens"] for item in found], [0, 10, 15])
        self.assertEqual([item["suffix_target_tokens"] for item in found], [25, 15, 10])
        # Answers of one length share their prompt, so a short true prefix may not tell them apart.
        model = Memorizer(examples, source.vocab)
        for item in prefix_extraction(model, examples, 2, "cpu"):
            exact = correct = 0
            for example in examples:
                given = min(len(example.answer) - 2, math.floor((len(example.answer) - 1) * item["prefix_fraction"]))
                suffix = example.answer[given:]
                generated = greedy(model.table, example.prompt + example.answer[:given], len(suffix))
                exact += generated == suffix
                correct += sum(map(int.__eq__, generated, suffix))
            self.assertEqual((item["suffix_exact_accuracy"], item["suffix_token_accuracy"]),
                             (exact / len(examples), correct / item["suffix_target_tokens"]))


class NoiseFitTests(unittest.TestCase):
    def test_a_model_of_the_noisy_labels_is_told_from_one_of_the_clean_ones(self):
        source = generator(Problem(Copy(symbols=4), 4, None))
        clean = build_split(source, "train", 0, 3)
        observed, noise = noisy_split(clean, 1, 4)
        rows = teacher_forced(observed.examples), teacher_forced(clean.examples)
        found = noise_fit(Memorizer(observed.examples, source.vocab), *rows, 2)
        self.assertEqual(found["corrupted_targets"], noise["changed_targets"])
        self.assertEqual((found["observed_label_accuracy"], found["clean_label_accuracy"]), (1, 0))
        # A model of the clean labels reads the corrupted answers too, which only the first answer token escapes.
        found = noise_fit(Memorizer(clean.examples, source.vocab), *rows, 2)
        self.assertEqual((found["observed_label_accuracy"], found["clean_label_accuracy"]),
                         (0, len(clean.examples) / noise["changed_targets"]))


class CorpusTests(unittest.TestCase):
    def test_disjoint_pools_share_no_input_and_repeated_rows_are_counted(self):
        source = generator(Problem(Copy(symbols=8), 4, None))
        pool = pools(source, 0, {"train": 9, "validation": 5, "test": 5}, disjoint=True)
        self.assertTrue(all(item["unique_inputs"] == 0 for item in corpus_report(pool)["overlaps"].values()))
        repeated = replace(pool["validation"], examples=(pool["train"].examples[0],) * 3)
        found = corpus_report({"train": pool["train"], "validation": repeated})
        self.assertEqual(found["splits"]["validation"]["duplicate_rows"], 2)
        self.assertEqual(found["overlaps"]["train/validation"]["rows_in_second"], 3)

    def test_the_conflict_floor_is_the_empirical_conditional_entropy(self):
        found = context_report([example_from_targets("lookup", (BOS,), (label,)) for label in (ACCEPT, ACCEPT, REJECT)])
        self.assertEqual(found["conflicting_contexts"], 1)
        self.assertAlmostEqual(found["empirical_min_token_error"], 1 / 3)
        self.assertAlmostEqual(found["empirical_min_token_loss_nats"], -(2 * math.log(2 / 3) + math.log(1 / 3)) / 3)


if __name__ == "__main__":
    unittest.main()
