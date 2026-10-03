"""Free generation: what each rollout generates from a prompt, and how a generation is scored.

The cases are those of `experiments/synthetic_trainers/tests/test_generation.py`.
Test doubles with exact logits make both rollouts write the same tokens, so
each case holds for both.
"""

import unittest

import torch
from torch import nn

from lab.domain.generative import Copy, Mode, Parity
from lab.domain.tasks import Problem
from lab.infrastructure.benchmarks.synthetic import generator
from lab.infrastructure.benchmarks.synthetic.generation import rollout, score, static_rollout
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.splits import build_split
from lab.infrastructure.benchmarks.synthetic.vocabulary import BOS, EOS, IDENTITY_BASE, PAD, SEP

ROLLOUTS = (rollout, static_rollout)


class CausalOracle(nn.Module):
    """A test double whose prediction at a position reads only the prefix up to it."""

    def __init__(self, source, wrong_first=False):
        super().__init__()
        self.source, self.wrong_first = source, wrong_first
        self.contexts = []

    def forward(self, tokens):
        self.contexts.append(tokens.tolist())
        logits = torch.zeros((*tokens.shape, self.source.vocab))
        for row in range(len(tokens)):
            for position in range(tokens.shape[1]):
                prefix = tokens[row, :position + 1].tolist()
                predicted = PAD
                if SEP in prefix:
                    end = prefix.index(SEP) + 1
                    answer, supplied = self.source.answer(tuple(prefix[:end])), tuple(prefix[end:])
                    if self.wrong_first and not supplied:
                        predicted = IDENTITY_BASE + (answer[0] - IDENTITY_BASE + 1) % self.source.task.symbols
                    elif supplied != answer[:len(supplied)] or len(supplied) >= len(answer):
                        predicted = EOS
                    else:
                        predicted = answer[len(supplied)]
                logits[row, position, predicted] = 30.0
        return logits


class Constant(nn.Module):
    def __init__(self, vocab, token):
        super().__init__()
        self.vocab, self.token = vocab, token

    def forward(self, tokens):
        logits = torch.zeros((*tokens.shape, self.vocab))
        logits[..., self.token] = 30.0
        return logits


def answers(source, seed, count):
    """A test split of `source`, and what its free generation reads."""
    examples = build_split(source, "test", seed, count).examples
    return examples, Rows(examples, "cpu")[0:count].answers


def generations(predictions, count):
    return [tuple(row[:size]) for row, size in zip(predictions.tolist(), count.tolist())]


def written(chunk, generated):
    """Generations as a rollout returns them, at least one token per step of the chunk's generation."""
    steps = max(len(chunk.widths), *map(len, generated))
    return (torch.tensor([list(row) + [PAD] * (steps - len(row)) for row in generated]),
            torch.tensor([len(row) for row in generated]))


def report(chunk, generated, final_token=False):
    """Correct tokens, exact answers, correct final answers, answers ended by EOS, tokens generated, extra tokens."""
    totals, _ = score(chunk, *written(chunk, generated), final_token)
    return totals.tolist()[:6]


class RolloutTests(unittest.TestCase):
    def test_a_perfect_causal_decoder_generates_complete_answers_of_every_length(self):
        source = generator(Problem(Copy(symbols=8), 5, 1))
        examples, chunk = answers(source, 2, 12)
        self.assertGreater(len({len(example.prompt) for example in examples}), 1)
        for generate in ROLLOUTS:
            with self.subTest(rollout=generate.__name__):
                predictions, count = generate(CausalOracle(source), chunk)
                self.assertEqual(generations(predictions, count), [example.answer for example in examples])
                tokens = sum(len(example.answer) for example in examples)
                self.assertEqual(report(chunk, generations(predictions, count)), [tokens, 12, 12, 12, tokens, 0])

    def test_a_rollout_reads_its_own_wrong_token_as_context(self):
        source = generator(Problem(Copy(symbols=8), 5, None))
        examples, chunk = answers(source, 0, 3)
        for generate in ROLLOUTS:
            with self.subTest(rollout=generate.__name__):
                model = CausalOracle(source, wrong_first=True)
                predictions, count = generate(model, chunk)
                for row, (example, generated) in enumerate(zip(examples, generations(predictions, count))):
                    self.assertNotEqual(generated[0], example.answer[0])
                    self.assertEqual(generated[1:], (EOS,))
                    self.assertEqual(model.contexts[1][row][len(example.prompt)], generated[0])

    def test_immediate_eos_ends_every_row_after_one_token(self):
        source = generator(Problem(Copy(symbols=8), 5, None))
        _, chunk = answers(source, 4, 3)
        for generate in ROLLOUTS:
            with self.subTest(rollout=generate.__name__):
                predictions, count = generate(Constant(source.vocab, EOS), chunk)
                self.assertEqual(generations(predictions, count), [(EOS,)] * 3)

    def test_without_eos_each_row_stops_at_its_own_limit(self):
        source = generator(Problem(Copy(symbols=8), 5, 1))
        examples, chunk = answers(source, 1, 8)
        for generate in ROLLOUTS:
            with self.subTest(rollout=generate.__name__):
                predictions, count = generate(Constant(source.vocab, BOS), chunk)
                self.assertEqual(count.tolist(), [example.generation_limit for example in examples])
                self.assertEqual(report(chunk, generations(predictions, count))[3], 0)


class ScoreTests(unittest.TestCase):
    def test_extra_output_is_not_an_exact_answer(self):
        source = generator(Problem(Copy(symbols=8), 5, None))
        (example,), chunk = answers(source, 3, 1)
        size = len(example.answer)
        self.assertEqual(report(chunk, [(*example.answer[:-1], BOS, EOS)]), [size - 1, 0, 0, 1, size + 1, 1])

    def test_missing_answer_tokens_are_errors(self):
        source = generator(Problem(Copy(symbols=8), 5, None))
        (example,), chunk = answers(source, 2, 1)
        self.assertEqual(report(chunk, [example.answer[:2]]), [2, 0, 0, 0, 2, 0])

    def test_a_final_answer_can_be_correct_after_a_wrong_scratchpad(self):
        for task in (Mode(symbols=8, scratchpad="counts"), Parity(scratchpad="running")):
            with self.subTest(task=task):
                source = generator(Problem(task, 6, None))
                (example,), chunk = answers(source, 0, 1)
                generated = (BOS, *example.answer[1:])
                self.assertEqual(report(chunk, [generated], final_token=True)[1:3], [0, 1])
                self.assertEqual(report(chunk, [generated[:-1]], final_token=True)[1:3], [0, 0])


if __name__ == "__main__":
    unittest.main()
