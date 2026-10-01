"""Verify the extracted AMSGradW against its Lean rule and frozen winner."""

import copy
import io
import json
import math
import os
from pathlib import Path
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch

from gpt_mini.amsgradw import AMSGradW
from gpt_mini.gpt_mini import Config, GPTMini


def cuda_snapshot():
    """Count actual CUDA Graph objects and Dynamo tracing in PyTorch 2.14."""
    from torch._dynamo.utils import counters
    from torch._inductor.cudagraph_trees import get_manager

    manager = get_manager(create_if_none_exists=False)
    seen = set()

    def visit(node):
        if id(node) in seen:
            return 0
        seen.add(id(node))
        return int(node.graph is not None) + sum(
            visit(child) for children in node.children.values() for child in children)

    nodes = sum(visit(root) for root in manager.get_roots()) if manager is not None else 0
    return {"unique_fx_graphs": counters["stats"]["unique_graphs"],
            "traced_calls": counters["stats"]["calls_captured"],
            "dynamo_frames": dict(counters["frames"]),
            "graph_breaks": dict(counters["graph_break"]), "recorded_graph_nodes": nodes}


class AMSGradWTests(unittest.TestCase):
    def test_actual_moments_maximum_and_old_weight_decay_match_scalar_rule(self):
        p = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
        opt = AMSGradW([p], lr=.05, betas=(.6, .7), eps=.2, weight_decay=.4)
        expected = [2., -3.]
        momentum, second, maximum = [0., 0.], [0., 0.], [0., 0.]
        for gradient in ([1., -2.], [-3., .2], [0., 0.]):
            for i, g in enumerate(gradient):
                momentum[i] = .6 * momentum[i] + .4 * g
                second[i] = .7 * second[i] + .3 * g * g
                maximum[i] = max(maximum[i], second[i])
                expected[i] = .98 * expected[i] - .05 * momentum[i] / (.2 + math.sqrt(maximum[i]))
            p.grad = torch.tensor(gradient, dtype=p.dtype)
            opt.step()
            torch.testing.assert_close(p, torch.tensor(expected, dtype=p.dtype), rtol=1e-13, atol=1e-13)
            for key, value in (("exp_avg", momentum), ("exp_avg_sq", second),
                               ("max_exp_avg_sq", maximum)):
                torch.testing.assert_close(opt.state[p][key], torch.tensor(value, dtype=p.dtype))

    def test_matches_the_frozen_benchmark_variant(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/amsgradw_benchmark.json").read_text())

        def tensor(values):
            return torch.tensor(values, dtype=torch.float64).reshape(fixture["shape"])

        for case in fixture["cases"]:
            p = torch.nn.Parameter(tensor(case["initial"]))
            opt = AMSGradW([p], lr=fixture["lr"], betas=tuple(fixture["betas"]),
                           eps=fixture["eps"], weight_decay=case["weight_decay"])
            for i, step in enumerate(case["steps"]):
                with self.subTest(decay=case["weight_decay"], step=i + 1):
                    p.grad = tensor(step["gradient"])
                    opt.step()
                    torch.testing.assert_close(p, tensor(step["weight"]), rtol=0, atol=0)
                    for new, original in (("exp_avg", "m"), ("exp_avg_sq", "v"),
                                           ("max_exp_avg_sq", "maximum")):
                        torch.testing.assert_close(opt.state[p][new], tensor(step[original]), rtol=0, atol=0)

    def test_shifted_quadratic_reproduces_lean_weighted_equilibrium(self):
        # AMSGradW.shiftedRun, eta1/4, epsilon1, decay2, beta9/10, beta2=1/2.
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = AMSGradW([p], lr=.25, betas=(.9, .5), eps=1., weight_decay=2.)
        for _ in range(1500):
            p.grad = p.detach() - 1
            opt.step()
        denominator = 1 + opt.state[p]["max_exp_avg_sq"].sqrt()
        torch.testing.assert_close(p, 1 / (1 + 2 * denominator), rtol=0, atol=1e-12)
        self.assertGreater((p - 1).abs().item(), .5)

    def test_groups_and_parameters_without_gradient(self):
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        idle = torch.nn.Parameter(p.detach().clone())
        opt = AMSGradW([{"params": [p], "lr": .1, "weight_decay": .3},
                        {"params": [q, idle], "lr": .05, "weight_decay": 0.}], eps=1.)
        p.grad, q.grad = torch.zeros_like(p), torch.zeros_like(q)
        opt.step()
        torch.testing.assert_close(p, torch.full_like(p, .97))
        torch.testing.assert_close(q, torch.ones_like(q))
        torch.testing.assert_close(idle, torch.ones_like(idle))
        self.assertTrue(all((v == 0).all().item() for v in opt.state[idle].values()))
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)
        self.assertIsNone(q.grad)

    def test_serialized_state_resumes_the_same_next_step(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        opt = AMSGradW([p])
        p.grad = torch.tensor([.1, -.3], dtype=p.dtype)
        opt.step()
        stream = io.BytesIO()
        torch.save({"weights": p.detach().clone(), "optimizer": opt.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        q = torch.nn.Parameter(checkpoint["weights"])
        resumed = AMSGradW([q], lr=.1)
        resumed.load_state_dict(checkpoint["optimizer"])
        p.grad = torch.tensor([-.7, .5], dtype=p.dtype)
        q.grad = p.grad.clone()
        opt.step()
        resumed.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in opt.state[p]:
            torch.testing.assert_close(opt.state[p][key], resumed.state[q][key], rtol=0, atol=0)

    def test_added_group_is_initialized_and_invalid_values_are_rejected(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = AMSGradW([p])
        q = torch.nn.Parameter(torch.ones(3))
        opt.add_param_group({"params": [q], "lr": .002})
        self.assertEqual(set(opt.state[q]), {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})
        for kw in ({"lr": -.1}, {"lr": math.nan}, {"eps": 0.}, {"eps": math.inf},
                   {"weight_decay": -.1}, {"betas": (1., .9)}, {"betas": (.9, 1.1)}):
            with self.subTest(options=kw), self.assertRaises(ValueError):
                AMSGradW([torch.nn.Parameter(torch.ones(1))], **kw)
        with self.assertRaises(ValueError):
            AMSGradW([p, p])
        with self.assertRaises(ValueError):
            opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))], "eps": 0.})
        self.assertEqual(len(opt.param_groups), 2)

    def test_sparse_gradient_is_rejected_before_any_weight_or_state_changes(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = AMSGradW([p])
        p.grad = torch.ones_like(p).to_sparse()
        with self.assertRaises(RuntimeError):
            opt.step()
        torch.testing.assert_close(p, torch.ones_like(p))
        self.assertTrue(all((v == 0).all().item() for v in opt.state[p].values()))

    def test_gptmini_tied_embedding_is_updated_once(self):
        torch.set_num_threads(1)
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        model = GPTMini(cfg)
        opt = AMSGradW(model.parameters())
        tokens = torch.arange(8).reshape(1, 8)
        loss = torch.nn.functional.cross_entropy(model(tokens).flatten(0, 1), tokens.roll(-1).flatten())
        loss.backward()
        initial, gradient = model.embed.weight.detach().clone(), model.embed.weight.grad.clone()
        opt.step()
        state = opt.state[model.embed.weight]
        expected = initial - .0003 * (.1 * gradient) / (1e-8 + (.001 * gradient.square()).sqrt()) - .000003 * initial
        torch.testing.assert_close(model.embed.weight, expected)
        self.assertIs(model.embed.weight, model.unembed.weight)
        self.assertEqual(sum(p is model.embed.weight for g in opt.param_groups for p in g["params"]), 1)


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class CudaTests(unittest.TestCase):
    def test_full_compiled_step_matches_eager_with_real_cuda_graphs(self):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.compiler.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        torch.manual_seed(42)
        model = GPTMini(cfg).to("cuda")
        eager = copy.deepcopy(model)
        opt, reference = AMSGradW(model.parameters()), AMSGradW(eager.parameters())
        parameters = dict(model.named_parameters())
        for p in parameters.values():
            p.grad = torch.zeros_like(p)
            torch._dynamo.mark_static_address(p, guard=True)
            torch._dynamo.mark_static_address(p.grad, guard=True)
        for state in opt.state.values():
            for value in state.values():
                torch._dynamo.mark_static_address(value, guard=True)
        for value in (model.cos, model.sin):
            torch._dynamo.mark_static_address(value, guard=True)
        tokens = torch.arange(8, device="cuda").reshape(1, 8)
        target = tokens.roll(-1, 1)

        def loss_fn(params):
            logits = torch.func.functional_call(model, params, (tokens,))
            return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())

        gradient_fn = torch.func.grad_and_value(loss_fn)

        def train():
            gradients, loss = gradient_fn(parameters)
            with torch.no_grad():
                for name, p in parameters.items():
                    p.grad.copy_(gradients[name])
                opt.step()
            return loss.detach()

        compiled = torch.compile(train, fullgraph=True, dynamic=False, mode="reduce-overhead")
        for i in range(20):
            reference.zero_grad(set_to_none=True)
            expected = torch.nn.functional.cross_entropy(eager(tokens).flatten(0, 1), target.flatten())
            expected.backward()
            reference.step()
            torch.compiler.cudagraph_mark_step_begin()
            actual = compiled()
            torch.testing.assert_close(actual, expected.detach(), rtol=2e-4, atol=2e-5)
            for p, q in zip(model.parameters(), eager.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-3, atol=4e-5)
                for key in opt.state[p]:
                    torch.testing.assert_close(opt.state[p][key], reference.state[q][key], rtol=1e-3, atol=4e-5)
            if i == 5:
                warm = cuda_snapshot()
        final = cuda_snapshot()
        self.assertEqual(final["graph_breaks"], {})
        self.assertGreater(final["recorded_graph_nodes"], 0)
        self.assertEqual(warm, final)
        del compiled, model, eager, opt, reference
        torch.compiler.reset()


if __name__ == "__main__":
    unittest.main()
