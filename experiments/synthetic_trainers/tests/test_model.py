"""The reference mini GPT remains causal, trainable, and correctly optimized."""

import unittest

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import masked_loss
from experiments.synthetic_trainers.records import collate
from experiments.synthetic_trainers.runtime import optimizer_for
from experiments.synthetic_trainers.specs import TaskSpec


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, vocab=49, length=64):
        torch.manual_seed(2)
        return GPTMini(ModelSpec(width=16, layers=2, heads=2).reference_config(vocab, length))

    def test_future_mutations_do_not_change_prefix_logits(self):
        model = self.model().eval()
        tokens = torch.randint(0, 49, (3, 24))
        changed = tokens.clone()
        changed[:, 9:] = torch.randint(0, 49, (3, 15))
        with torch.no_grad():
            first, second = model(tokens), model(changed)
        torch.testing.assert_close(first[:, :9], second[:, :9], rtol=1e-5, atol=1e-5)

    def test_batch_padding_does_not_change_valid_prefix_predictions(self):
        spec = TaskSpec(length=48, min_length=24, pairs=4, queries=2)
        examples = build_split(spec, "train", 1, 5).examples
        model = self.model(spec.vocab_size).eval()
        with torch.no_grad():
            batched = model(collate(examples).tokens)
            for row, example in enumerate(examples):
                alone = model(collate([example]).tokens)
                torch.testing.assert_close(batched[row, :len(example.tokens)], alone[0], rtol=1e-5, atol=1e-5)

    def test_optimizer_covers_tied_embeddings_once_and_does_not_decay_temperature(self):
        model = self.model()
        self.assertIs(model.embed.weight, model.unembed.weight)
        config = TrainConfig()
        optimizer = optimizer_for(model, config)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        self.assertEqual(sum(p is model.embed.weight for p in parameters), 1)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertEqual(group["weight_decay"], config.weight_decay if parameter.ndim >= 2 else 0)

    def test_backward_updates_embedding_attention_and_feed_forward_parameters(self):
        spec = TaskSpec(length=24, pairs=4, queries=2)
        model = self.model(spec.vocab_size)
        batch = collate(build_split(spec, "train", 0, 8).examples)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = optimizer_for(model, TrainConfig(learning_rate=0.01))
        masked_loss(model(batch.tokens), batch.targets).backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertFalse(torch.equal(parameter, before[name]))

    def test_actual_optimization_reduces_loss_on_prefix_data(self):
        spec = TaskSpec(task="blocks", length=16, blocks=2, neutral_fraction=0.2)
        batch = collate(build_split(spec, "train", 5, 16).examples)
        model = self.model(spec.vocab_size, 16)
        optimizer = optimizer_for(model, TrainConfig(learning_rate=0.01))
        with torch.no_grad():
            initial = float(masked_loss(model(batch.tokens), batch.targets))
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(batch.tokens), batch.targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        with torch.no_grad():
            final = float(masked_loss(model(batch.tokens), batch.targets))
        self.assertLess(final, initial * 0.5)

    def test_invalid_model_and_training_configurations_fail_early(self):
        for fields in (dict(width=15, heads=2), dict(width=12, heads=4), dict(layers=0),
                       dict(rope_theta=float("inf"))):
            with self.assertRaises(ValueError):
                ModelSpec(**fields)
        for fields in (dict(steps=0), dict(target=1.1), dict(target_metric="test"),
                       dict(eval_lengths=(64, 64)), dict(learning_rate=float("nan"))):
            with self.assertRaises(ValueError):
                TrainConfig(**fields)
