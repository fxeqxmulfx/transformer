"""Real CUDA parity and stable complete graphs before benchmark training."""

import gc
import os
import unittest

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import training_starts, windows
from gpt_mini.infrastructure.benchmark.compiled_benchmark.telemetry import snapshot
from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.optimizer import ExtensionOptimizer
from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.step import ExtensionStep


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class CudaTests(unittest.TestCase):
    def test_updates_all_states_windows_graph_stability_and_reload(self):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = Config(vocab_size=8,n_layers=1,n_heads=2,d_model=16,d_ff=32,max_seq_len=8)
        tokens = (torch.arange(400) % 8).to("cuda")
        starts = training_starts(400,8,2,60,0).to("cuda")
        for attention in ("softmax", "sparsemax"):
            for method in ("amsgradw", "amsgradmd", "amsgradmd_guarded"):
                with self.subTest(attention=attention, method=method):
                    torch.compiler.reset()
                    from torch._dynamo.utils import counters
                    counters.clear()
                    eager = make_model(cfg,attention,0,"cuda")
                    model = make_model(cfg,attention,0,"cuda")
                    rate = .3 if method.endswith("guarded") else .001
                    opt = ExtensionOptimizer(dict(eager.named_parameters()),method,rate)
                    runtime = ExtensionStep(model,attention,method,rate,0,2,60)
                    for index in range(4):
                        inputs, targets = windows(tokens,starts[index],8)
                        eager.zero_grad(set_to_none=True)
                        expected = F.cross_entropy(eager(inputs).flatten(0,1),targets.flatten())
                        expected.backward()
                        loss = runtime.train(tokens,starts)
                        torch.testing.assert_close(loss,expected.detach(),rtol=2e-4,atol=2e-5)
                        gradients = {name:p.grad for name,p in eager.named_parameters()}
                        for name,p in runtime.parameters.items():
                            torch.testing.assert_close(p.grad,gradients[name],rtol=1e-3,atol=4e-5)
                        opt.step(gradients)
                        for name,p in runtime.parameters.items():
                            torch.testing.assert_close(p,opt.parameters[name],rtol=1e-3,atol=4e-5)
                        def compare(a,b):
                            if isinstance(a,dict):
                                self.assertEqual(set(a),set(b))
                                for key in a: compare(a[key],b[key])
                            else:
                                torch.testing.assert_close(a,b,rtol=1e-3,atol=4e-5)
                        compare(runtime.opt.state,opt.state)
                        torch.testing.assert_close(runtime.opt.accepted,opt.accepted)
                    for _ in range(4): runtime.train(tokens,starts)
                    for _ in range(2): runtime.evaluate(tokens[:80],2)
                    warm = snapshot()
                    self.assertEqual(warm["graph_breaks"],{})
                    self.assertGreater(warm["recorded_graph_nodes"],0)
                    saved = {name:p.detach().cpu().clone() for name,p in model.state_dict().items()}
                    expected_val = runtime.evaluate(tokens[:80],2)
                    for _ in range(40): runtime.train(tokens,starts)
                    runtime.evaluate(tokens[:80],2)
                    self.assertEqual(warm,snapshot())
                    model.load_state_dict(saved)
                    self.assertEqual(runtime.evaluate(tokens[:80],2),expected_val)
                    self.assertIs(model.embed.weight,model.unembed.weight)
                    runtime.close()
                    del runtime,opt,model,eager
                    torch.compiler.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
