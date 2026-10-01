"""Free decoding, variable prompts, EOS accounting, and exposure to own mistakes."""

from dataclasses import replace
import unittest

import torch
from torch import nn

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.generation import Rollout, rollout
from experiments.synthetic_trainers.generation_metrics import GenerationMetrics
from experiments.synthetic_trainers.metrics import evaluate, masked_loss
from experiments.synthetic_trainers.records import collate, generation_example
from experiments.synthetic_trainers.runtime import optimizer_for
from experiments.synthetic_trainers.sequence_oracles import generation_answer
from experiments.synthetic_trainers.specs import TaskSpec


class CausalOracle(nn.Module):
    """A test double whose prediction reads only the current prefix."""
    def __init__(self, spec, wrong_first=False):
        super().__init__()
        self.spec, self.wrong_first = spec, wrong_first
        self.contexts = []

    def forward(self, tokens):
        self.contexts.append(tokens.tolist())
        logits = torch.zeros((*tokens.shape, self.spec.vocab_size), device=tokens.device)
        for row in range(len(tokens)):
            for position in range(tokens.shape[1]):
                prefix = tokens[row, :position + 1].tolist()
                predicted = v.PAD
                if v.SEP in prefix:
                    separator = prefix.index(v.SEP)
                    answer = generation_answer(prefix[:separator + 1], self.spec)
                    supplied = tuple(prefix[separator + 1:])
                    index = len(supplied)
                    if self.wrong_first and not supplied:
                        predicted = v.IDENTITY_BASE + (answer[0] - v.IDENTITY_BASE + 1) % self.spec.symbols
                    elif supplied != answer[:index] or index >= len(answer):
                        predicted = v.EOS
                    else:
                        predicted = answer[index]
                logits[row, position, predicted] = 30.0
        return logits


class ConstantModel(nn.Module):
    def __init__(self, vocab, token, nonfinite=False):
        super().__init__()
        self.vocab, self.token, self.nonfinite = vocab, token, nonfinite

    def forward(self, tokens):
        logits = torch.zeros((*tokens.shape, self.vocab), device=tokens.device)
        logits[..., self.token] = 30.0
        return logits * float("nan") if self.nonfinite else logits


def result_for(*predictions):
    rows = tuple(tuple(row) for row in predictions)
    return Rollout(rows, tuple(row[-1] == v.EOS for row in rows), 1, 1, 1, 0.0)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def spec(self, **fields):
        return TaskSpec(task="copy", length=5, symbols=8, **fields)

    def test_perfect_causal_decoder_generates_complete_variable_length_answers(self):
        spec = self.spec(min_length=1)
        examples = build_split(spec, "test", 2, 12).examples
        model = CausalOracle(spec).train()
        result = rollout(model, [row.prompt for row in examples], [row.generation_limit for row in examples])
        self.assertEqual(result.predictions, tuple(row.answer for row in examples))
        self.assertTrue(all(result.terminated))
        self.assertTrue(model.training)
        self.assertGreater(len({len(row.prompt) for row in examples}), 1)
        self.assertEqual(result.forward_calls, max(len(row.answer) for row in examples))
        self.assertLess(len(model.contexts[-1]), len(examples))  # Completed rows are removed.

    def test_rollout_feeds_its_own_wrong_token_into_the_next_context(self):
        spec = self.spec()
        row = build_split(spec, "test", 0, 1).examples[0]
        model = CausalOracle(spec, wrong_first=True)
        result = rollout(model, [row.prompt], [row.generation_limit])
        self.assertNotEqual(result.predictions[0][0], row.answer[0])
        self.assertEqual(result.predictions[0][-1], v.EOS)
        self.assertEqual(model.contexts[1][0][-1], result.predictions[0][0])
        self.assertNotEqual(model.contexts[1][0][-1], row.answer[0])

    def test_teacher_forced_accuracy_does_not_replace_free_generation_scores(self):
        spec = self.spec()
        examples = build_split(spec, "test", 0, 4).examples
        report = evaluate(CausalOracle(spec, wrong_first=True), examples, 3, spec=spec)
        self.assertGreater(report["teacher_forced"]["token_accuracy"], 0.8)
        self.assertEqual(report["sequence_accuracy"], 0.0)
        self.assertEqual(report["token_accuracy"], 0.0)
        self.assertEqual(report["eos_rate"], 1.0)

    def test_immediate_eos_is_an_incorrect_answer_and_stops_after_one_call(self):
        spec = self.spec()
        rows = build_split(spec, "test", 4, 3).examples
        report = evaluate(ConstantModel(spec.vocab_size, v.EOS), rows, 3, spec=spec)
        self.assertEqual(report["sequence_accuracy"], 0.0)
        self.assertEqual(report["token_accuracy"], 0.0)
        self.assertEqual(report["eos_rate"], 1.0)
        self.assertEqual(report["generated_tokens"], 3)
        self.assertEqual(report["generation_forward_calls"], 1)

    def test_never_eos_is_bounded_independently_for_each_prompt(self):
        spec = self.spec(min_length=1)
        rows = build_split(spec, "test", 1, 8).examples
        result = rollout(ConstantModel(spec.vocab_size, v.BOS), [row.prompt for row in rows],
                         [row.generation_limit for row in rows])
        self.assertFalse(any(result.terminated))
        self.assertEqual(tuple(map(len, result.predictions)), tuple(row.generation_limit for row in rows))
        self.assertEqual(result.forward_calls, max(row.generation_limit for row in rows))

    def test_batched_and_individual_rollouts_have_identical_answers(self):
        spec = self.spec(min_length=1)
        rows = build_split(spec, "test", 12, 7).examples
        model = CausalOracle(spec)
        together = rollout(model, [row.prompt for row in rows], [row.generation_limit for row in rows])
        separate = [rollout(model, [row.prompt], [row.generation_limit]).predictions[0] for row in rows]
        self.assertEqual(together.predictions, tuple(separate))
        report = evaluate(model, rows, 3, spec=spec)
        self.assertEqual(report["sequence_accuracy"], 1.0)
        self.assertEqual(report["target_count"], sum(len(row.answer) for row in rows))
        self.assertEqual(report["generated_tokens"], report["target_count"])

    def test_extra_output_cannot_count_as_an_exact_answer(self):
        spec = self.spec()
        row = build_split(spec, "test", 3, 1).examples[0]
        metrics = GenerationMetrics(spec)
        metrics.add([row], result_for((*row.answer[:-1], v.BOS, v.EOS)))
        report = metrics.report()
        self.assertEqual(report["sequence_accuracy"], 0.0)
        self.assertEqual(report["extra_tokens"], 1)
        self.assertLess(report["token_accuracy"], 1.0)

    def test_short_output_counts_missing_answer_tokens_as_errors(self):
        spec = self.spec()
        row = build_split(spec, "test", 2, 1).examples[0]
        metrics = GenerationMetrics(spec)
        metrics.add([row], result_for(row.answer[:2]))
        report = metrics.report()
        self.assertEqual(report["correct"], 2)
        self.assertEqual(report["target_count"], len(row.answer))
        self.assertEqual(report["token_accuracy"], 2 / len(row.answer))
        self.assertEqual(report["eos_rate"], 0.0)

    def test_final_answer_can_be_correct_even_when_scratchpad_is_wrong(self):
        for task, scratchpad in (("mode", "counts"), ("parity", "running")):
            spec = TaskSpec(task=task, length=6, symbols=8, scratchpad=scratchpad)
            row = build_split(spec, "test", 0, 1).examples[0]
            prediction = (v.BOS, *row.answer[1:])
            metrics = GenerationMetrics(spec)
            metrics.add([row], result_for(prediction))
            report = metrics.report()
            self.assertEqual(report["sequence_accuracy"], 0.0)
            self.assertEqual(report["final_answer_accuracy"], 1.0)

    def test_nonfinite_decoding_restores_original_training_mode(self):
        model = ConstantModel(64, v.BOS, nonfinite=True).train()
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            rollout(model, [(v.BOS, v.SEP)], [3])
        self.assertTrue(model.training)

    def test_invalid_limits_empty_prompts_and_mixed_supervision_fail(self):
        model = ConstantModel(64, v.EOS)
        for prompts, limits in (([], []), ([()], [1]), ([(v.BOS,)], []), ([(v.BOS,)], [0])):
            with self.assertRaises(ValueError):
                rollout(model, prompts, limits)
        with self.assertRaises(ValueError):
            GenerationMetrics().report()
        copy = build_split(self.spec(), "test", 0, 1).examples[0]
        prefix = build_split(TaskSpec(task="blocks", length=12), "test", 0, 1).examples[0]
        model.train()
        with self.assertRaises(ValueError):
            evaluate(model, [copy, prefix], 2)
        self.assertTrue(model.training)

    def test_training_examples_reject_unshifted_inputs_and_multiple_eos(self):
        spec = self.spec()
        row = build_split(spec, "test", 2, 1).examples[0]
        with self.assertRaises(ValueError):
            replace(row, tokens=(*row.tokens[:-1], row.answer[-1]))
        with self.assertRaises(ValueError):
            generation_example("copy", row.prompt, (v.EOS, v.EOS), 2)

    def test_real_mini_gpt_learns_a_small_autoregressive_copy_problem(self):
        torch.manual_seed(4)
        spec = TaskSpec(task="copy", length=2, symbols=2)
        rows = build_split(spec, "train", 1, 8).examples
        batch = collate(rows)
        model = GPTMini(ModelSpec(width=16, layers=2, heads=2).reference_config(spec.vocab_size, spec.context_length))
        optimizer = optimizer_for(model, TrainConfig(learning_rate=0.01))
        with torch.no_grad():
            initial = float(masked_loss(model(batch.tokens), batch.targets))
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(batch.tokens), batch.targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        report = evaluate(model, rows, 3, spec=spec)
        self.assertLess(report["loss"], initial * 0.1)
        self.assertEqual(report["sequence_accuracy"], 1.0)
