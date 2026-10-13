"""Independent contraction, actual tensor/loss coupling and semantic controls.

Source: Structured.TensorBasisModel/Training at b0a43a8. Tests evaluate
arbitrary parameters and literal pair sums, not just the capability witness.
"""

import itertools
import math
import unittest

import torch
from torch.nn import functional as F

from lab.dsl import Eager, TensorStack, basis, swap
from lab.domain.generative import Parity
from lab.domain.tasks import AlternatingBlocks, MQAR
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.tensor import complete_labels, depth_step, parity_step
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.tensor import digits
from lab.infrastructure.nn.tensor_heads import TensorHeads, recover


def direct_pointer(head, fields, position, query):
    """Independent literal sum over every visible pair and small channels."""
    q = fields[query, :16].view(4, 4)
    terms, values = [], []
    for key, value in itertools.product(range(query + 1), repeat=2):
        k, v = fields[key, 16:32].view(4, 4), fields[value, 32:52].view(5, 4)
        bias = position[value] + head.chronology * value + head.relative[value - key + head.context - 1]
        terms.append(bias + (q + k).logsumexp(-1).sum() + v.logsumexp(-1).sum())
        values.append(v.softmax(-1))
    logits = torch.stack(terms)
    return logits.logsumexp(0), torch.einsum("j,jhd->hd", logits.softmax(0), torch.stack(values))


def direct_state(head, fields):
    """Sum all short state paths instead of multiplying endpoint matrices."""
    paths, mass = [], []
    length = len(fields)
    for states in itertools.product(range(6), repeat=length + 1):
        weight = head.initial.softmax(-1)[states[0]]
        for position in range(length):
            row = fields[position, :36].view(6, 6)[states[position]]
            weight = weight * row.softmax(-1)[states[position + 1]]
        paths.append(head.emission[states[-1]].softmax(-1))
        mass.append(weight)
    return torch.einsum("j,jhd->hd", torch.stack(mass), torch.stack(paths))


class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_prefix_contraction_matches_literal_all_pair_sum_and_gradients(self):
        head = TensorHeads(5, .2).double()
        fields = torch.randn(2, 4, 52, dtype=torch.float64, requires_grad=True)
        positions = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        head.chronology.data.fill_(1.3)
        head.relative.data.normal_()
        partition = head.pointer_terms(fields, positions)[0]
        means = head.pointer_channels(fields, positions)
        for batch in range(2):
            for query in range(4):
                logz, channels = direct_pointer(head, fields[batch], positions[batch], query)
                torch.testing.assert_close(partition[batch, query], logz, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(means[batch, query], channels, rtol=1e-12, atol=1e-12)
        arguments = (fields, positions, head.relative, head.chronology)
        actual = torch.autograd.grad(partition[0, -1] + means[0, -1].square().sum(), arguments, retain_graph=True)
        logz, channels = direct_pointer(head, fields[0], positions[0], 3)
        expected = torch.autograd.grad(logz + channels.square().sum(), arguments)
        for first, second in zip(actual, expected, strict=True):
            torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)

    def test_state_contraction_matches_whole_path_probability(self):
        head = TensorHeads(3, .3).double()
        fields = torch.randn(1, 2, 52, dtype=torch.float64)
        channels = head.state_channels(fields)[0]
        for query in range(2):
            torch.testing.assert_close(channels[query], direct_state(head, fields[0, :query + 1]))
        torch.testing.assert_close(channels.sum(-1), torch.ones(2, 5, dtype=torch.float64))

    def test_free_parameter_count_and_actual_prenorm_tied_stack(self):
        for width, depth, vocab, context in ((64, 2, 36, 128), (128, 6, 548, 64), (64, 2, 68, 19)):
            spec = TensorStack(width, depth, context)
            model = build_model(spec, vocab, 0).double()
            self.assertEqual(sum(p.numel() for p in model.parameters()), spec.parameter_count(vocab))
            self.assertTrue(all(block.attention.heads is model.heads for block in model.blocks))
            tokens = torch.tensor([[1, 9, 10, 11]])
            raw = model.inputs(tokens)
            normalized = F.rms_norm(raw, (width,), eps=spec.eps)
            fields, pos = recover(normalized)
            torch.testing.assert_close(fields, model.embed.fields[tokens], rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(pos, model.absolute[:4].expand(1, 4))
            coords = model.heads(normalized)
            hidden = raw
            for block in model.blocks:
                hidden = block(hidden)
            torch.testing.assert_close(hidden, F.pad(coords, (52, width - 62)), rtol=1e-12, atol=1e-12)
            expected = F.linear(F.rms_norm(F.pad(coords, (52, width - 62)), (width,), eps=spec.eps),
                                model.embed.weight)
            torch.testing.assert_close(model(tokens), expected, rtol=1e-11, atol=1e-11)
            for query in range(4):
                torch.testing.assert_close(model(tokens)[0, query], model(tokens[:, :query + 1])[0, -1])

    def test_complete_state_loss_and_gradients_equal_affine_raw_likelihood(self):
        model = build_model(TensorStack(64, 3, 19), 68, 0).double()
        tokens = torch.tensor([[1, 22, 21, 18, 25]])
        targets = torch.tensor([[-100, -100, -100, 25, 17]])
        observed = complete_labels(tokens, targets, Parity(), 68)
        actual = model.complete_losses(tokens, observed, 0)
        prev, state, target = observed.unbind(1)
        fields = model.embed.fields[tokens][..., :36].reshape(1, 5, 6, 6)
        row = fields.gather(2, prev[..., None, None].expand(-1, -1, 1, 6)).squeeze(2)
        transition = row.logsumexp(-1) - row.gather(-1, state[..., None]).squeeze(-1)
        emission = model.heads.emission[state]
        value = emission.logsumexp(-1).sum(-1) - emission.gather(-1, digits(target)[..., None]).squeeze(-1).sum(-1)
        expected = (model.heads.branch.logsumexp(-1) - model.heads.branch[0]
                    + model.heads.initial.logsumexp(-1) - model.heads.initial[0]
                    + transition.cumsum(1) + value)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        parameters = tuple(model.parameters())
        ga = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
        ge = torch.autograd.grad(expected.sum(), parameters, allow_unused=True)
        for a, e in zip(ga, ge, strict=True):
            torch.testing.assert_close(a, torch.zeros_like(a) if e is None else e, rtol=1e-10, atol=1e-11)

    def test_raw_labels_include_depth_order_latest_overwrite_and_parity_eos(self):
        self.assertEqual([depth_step(x, 0) for x in (1, 9, 10, 11)], [0, 1, 5, 0])
        x = torch.tensor([[1, 9, 9, 10, 10], [1, 9, 10, 9, 10]])
        labels = complete_labels(x, torch.zeros_like(x), AlternatingBlocks(blocks=2), 36)
        self.assertEqual(labels[:, 1, -1].tolist(), [2, 4])
        tokens = [1]
        for key in range(8):
            tokens.extend((36 + key, 292 + key))
        for key in range(8):
            tokens.extend((36 + key, 300 + key))
        tokens.extend((292, 36))
        x = torch.tensor([tokens])
        labels = complete_labels(x, torch.zeros_like(x), MQAR(symbols=256, pairs=16, overwrites=8, queries=8), 548)
        self.assertEqual(labels[0, :2, -1].tolist(), [17, 18])
        self.assertEqual(parity_step(25, parity_step(18, parity_step(22, 0))), 4)
        x = torch.tensor([[1, 22, 18, 25]])
        labels = complete_labels(x, torch.tensor([[-100, -100, 25, 17]]), Parity(), 68)
        self.assertEqual(labels[0, 1].tolist(), [0, 1, 3, 4])

    def test_actual_pointer_loss_is_negative_log_of_the_same_joint_and_jointly_convex(self):
        model = build_model(TensorStack(64, 2, 64), 548, 0).double()
        tokens = torch.tensor([[1, *itertools.chain.from_iterable((36 + k, 292 + k) for k in range(8)), 36]])
        target = torch.full_like(tokens, -100)
        target[0, -1] = 292
        task = MQAR(symbols=256, pairs=8, queries=8)
        observed = complete_labels(tokens, target, task, 548)
        actual = model.complete_losses(tokens, observed, 1)[0, -1]
        fields, position = model.embed.fields[tokens][0], model.absolute[:tokens.shape[1]]
        logz, _ = direct_pointer(model.heads, fields, position, tokens.shape[1] - 1)
        qk = (fields[-1, :16] + fields[1, 16:32]).view(4, 4)
        matching = qk[:, 0].sum()
        v = fields[2, 32:52].view(5, 4)
        energy = matching + v.gather(-1, digits(torch.tensor(292))[:, None]).sum()
        energy = energy + position[2] + 2 * model.heads.chronology + model.heads.relative[64]
        expected = -model.heads.branch.log_softmax(-1)[1] + logz - energy
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(actual, parameters)
        for offset in (0, 16, 32):
            self.assertGreater(float(gradients[1][:, offset:offset + 16].norm()), 0)
        endpoints = [[torch.randn_like(p) * .3 for p in parameters] for _ in range(2)]
        def value(fraction):
            with torch.no_grad():
                for p, a, b in zip(parameters, *endpoints, strict=True):
                    p.copy_((1 - fraction) * a + fraction * b)
                return float(model.complete_losses(tokens, observed, 1)[0, -1])
        first, middle, last = value(0), value(.37), value(1)
        self.assertLessEqual(middle, .63 * first + .37 * last + 1e-10)

    def test_finite_lean_state_witness_decodes_depth_order_and_both_parity_calls(self):
        for task, vocab, context in ((AlternatingBlocks(blocks=2), 36, 128), (Parity(), 68, 19)):
            model = build_model(TensorStack(64, 2, context), vocab, 0)
            step = depth_step if isinstance(task, AlternatingBlocks) else parity_step
            labels = ([15, 15, 16, 15, 15, 15] if isinstance(task, AlternatingBlocks)
                      else [0, 0, 24, 25, 17, 0])
            gain = math.log(100_000)
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
                for token, previous in itertools.product(range(vocab), range(6)):
                    model.embed.fields[token, 6 * previous + step(token, previous)] = gain
                model.heads.initial[0] = gain
                model.heads.branch[0] = gain
                model.heads.emission.scatter_(-1, digits(torch.tensor(labels))[..., None], gain)
            if isinstance(task, AlternatingBlocks):
                self.assertEqual(model.integer_function([1, 9, 9, 10, 10])[-1], 16)
                self.assertEqual(model.integer_function([1, 9, 10, 9, 10])[-1], 15)
            else:
                answer = model.integer_function([1, 22, 21, 18])
                self.assertEqual(answer[-1], 25)
                self.assertEqual(model.integer_function(answer)[-1], 17)

    def test_large_finite_recall_witness_stays_numerically_stable_and_uses_latest_write(self):
        model = build_model(TensorStack(128, 6, 64), 548, 0)
        gain, large = math.log(10 ** 12), 65 * math.log(10 ** 12)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            symbols = (torch.arange(548) + 220) % 256
            for start, assignment in ((0, digits(symbols, 4)), (16, digits(symbols, 4)),
                                      (32, digits(torch.arange(548)))):
                count = assignment.shape[-1]
                model.embed.fields[:, start:start + 4 * count].view(548, count, 4).scatter_(
                    -1, assignment[..., None], large)
            model.absolute.fill_(-large)
            model.absolute[2:33:2] = 0
            model.heads.relative.fill_(-large)
            model.heads.relative[64] = 0
            model.heads.chronology.fill_(gain)
            model.heads.branch[1] = gain
        tokens = [1]
        for repeat in range(2):
            for key in range(8):
                tokens.extend((36 + key, 292 + 8 * repeat + key))
        tokens.extend((547, 36))
        self.assertEqual(model.integer_function(tokens)[-1], 300)
        x = torch.tensor([tokens])
        target = torch.full_like(x, -100)
        target[0, -1] = 300
        observed = complete_labels(x, target, MQAR(symbols=256, pairs=16, overwrites=8, queries=8), 548)
        loss = model.complete_losses(x, observed, 1)[0, -1]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checked_integer_interface_and_selection_positions(self):
        model = build_model(TensorStack(64, 2, 4), 36, 0)
        tokens = torch.tensor([[1, 9, 10]])
        positions = torch.tensor([[0, 2]])
        torch.testing.assert_close(model(tokens, positions), model(tokens)[:, [0, 2]])
        self.assertEqual(model.integer_function([1, 9, 10]), [1, 9, 10, int(model(tokens)[0, -1].argmax())])
        for invalid in ([], [-1], [36], [1] * 5):
            self.assertEqual(model.integer_function(invalid), [*invalid, 0])

    def test_task_uses_identical_raw_splits_and_excludes_auxiliary_labels_from_forward(self):
        run = basis(TensorStack(64, 2, 64), "easy", execution=Eager(device="cpu"))["recall"]
        run = swap(swap(swap(run, "benchmark.train", 8), "benchmark.validation", 4), "benchmark.test", 4)
        complete = build_task(run.benchmark, 1, torch.device("cpu"), run.model)
        original = build_task(run.benchmark, 1, torch.device("cpu"))
        self.assertEqual(complete.fingerprints, original.fingerprints)
        model = build_model(run.model, complete.vocab, 0)
        batch = complete.inputs(torch.tensor([0, 1]), static=True)
        losses, targets = complete.forward(model, batch, supervised=True)
        self.assertTrue(torch.isfinite(complete.loss(losses, targets)))
        logits = model(batch[:, 0])
        altered = batch.clone()
        altered[:, 3:] = 0
        torch.testing.assert_close(model(altered[:, 0]), logits)
        self.assertFalse(torch.allclose(complete.forward(model, altered)[0], complete.forward(model, batch)[0]))


if __name__ == "__main__":
    unittest.main()
