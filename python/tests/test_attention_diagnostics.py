"""Fixed-row attention measurements: EXPERIMENT_PLAN.md, step 2.

Hand-made causal rows check the statistics independently of a model. A
rewritten MQAR table checks the identity of an answer route rather than
token matching. Compiled controls check complete training noninterference
and checkpointed turnover, including records beyond an interrupted save.
"""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lab.domain.experiment import Experiment, require_continuation
from lab.domain.model import PerHeadQKV, Softmax, Sparsemax
from lab.domain.optimizers import AdamW
from lab.domain.spec import describe, substitute, swap
from lab.domain.synthetic import Synthetic
from lab.domain.tasks import MQAR
from lab.domain.training import AttentionDiagnostics, Budget, Checkpoint, Compiled, Diagnostics, Evaluate, Schedule, Seeds
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.vocabulary import BOS, IDENTITY_BASE, IGNORE
from lab.infrastructure.engine.attention import AttentionObserver
from lab.infrastructure.engine.attention_batch import answer_routes, recall_queries
from lab.infrastructure.engine.attention_stats import statistics
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.attention import softmax
from lab.infrastructure.nn.sparsemax import causal_sparsemax
from lab.infrastructure.store import RunDirectory

from examples import gptmini, modular
from test_engine import Interrupted, untimed
from test_graphs import records, summary, train


def recall():
    """A small rewritten recall benchmark with variable padding and two answers per row."""
    benchmark = Synthetic(task=MQAR(symbols=8, pairs=4, queries=2, overwrites=2), length=17, min_length=12,
                          train=24, validation=16, test=16, target=.99)
    model = swap(gptmini(16, 2, 2), "context", 17)
    return Experiment(model=model, benchmark=benchmark, optimizer=AdamW(lr=1e-3, betas=(.9, .98), weight_decay=.1),
                      schedule=Schedule(),
                      budget=Budget(updates=20, batch=8), seeds=Seeds(model=3, data=1),
                      evaluate=Evaluate(every=5, batch=8), execution=Compiled(),
                      diagnostics=Diagnostics(every=5), checkpoint=Checkpoint(every=10))


class AttentionStatisticTests(unittest.TestCase):
    def test_hand_made_rows_check_every_statistic_and_turnover(self):
        scores = torch.tensor([[0., 99., 99., 99.], [2., 0., 99., 99.],
                               [0., 0., 2., 99.], [0., 0., 0., 0.]], dtype=torch.float64)[None, None]
        weights = causal_sparsemax(scores)
        lengths = torch.tensor([4])
        supervised = torch.tensor([[False, False, False, True]])
        actual, support = statistics(scores, weights, lengths, supervised)
        head = actual["all_rows"][0]
        expected = {"row_count": 4, "visible_pair_count": 10, "mean_support_positions": 7 / 4,
                    "mean_support_share": 17 / 24, "exact_zero_pair_share": .3,
                    "singleton_row_share": .75, "self_only_row_share": .5,
                    "score_gap_gt_one_share": .75, "normalized_entropy": .25,
                    "mean_score_std": (1 + math.sqrt(8 / 9)) / 4}
        for name, value in expected.items():
            with self.subTest(statistic=name):
                self.assertAlmostEqual(head[name], value, places=7)
        self.assertIsNone(head["turnover_share"])
        self.assertIsNone(head["turnover_changed_pairs"])
        nontrivial = actual["nontrivial_rows"][0]
        self.assertEqual(nontrivial["row_count"], 3)
        self.assertAlmostEqual(nontrivial["self_only_row_share"], 1 / 3)
        self.assertAlmostEqual(nontrivial["normalized_entropy"], 1 / 3)
        readout = actual["supervised_rows"][0]
        self.assertEqual(readout["row_count"], 1)
        self.assertEqual(readout["mean_support_positions"], 4)
        self.assertEqual(readout["normalized_entropy"], 1)
        previous = support.clone()
        previous[0, 0, 1, :2] = torch.tensor([False, True])
        previous[..., 0, 3] = True  # A masked future pair never enters turnover.
        changed, _ = statistics(scores, weights, lengths, supervised, previous)
        self.assertEqual(changed["all_rows"][0]["turnover_changed_pairs"], 2)
        self.assertAlmostEqual(changed["all_rows"][0]["turnover_share"], .2)
        self.assertEqual(changed["supervised_rows"][0]["turnover_changed_pairs"], 0)

    def test_softmax_entropy_and_sparsemax_shadow_have_different_supports(self):
        scores = torch.tensor([[0., 7.], [2., 0.]], dtype=torch.float64)[None, None]
        arguments = torch.tensor([2]), torch.tensor([[False, True]])
        dense, _ = statistics(scores, softmax(scores), *arguments)
        sparse, _ = statistics(scores, causal_sparsemax(scores), *arguments)
        head = dense["all_rows"][0]
        self.assertEqual(head["mean_support_share"], 1)
        self.assertEqual(head["exact_zero_pair_share"], 0)
        probability = math.exp(2) / (math.exp(2) + 1)
        entropy = -(probability * math.log(probability) + (1 - probability) * math.log(1 - probability))
        self.assertAlmostEqual(head["normalized_entropy"], entropy / (2 * math.log(2)), places=12)
        self.assertEqual(sparse["supervised_rows"][0]["singleton_row_share"], 1)
        self.assertEqual(dense["supervised_rows"][0]["singleton_row_share"], 0)

    def test_padding_and_structural_singletons_are_explicit(self):
        scores = torch.zeros(2, 2, 4, 4)
        weights = causal_sparsemax(scores)
        weights[1, :, 2:] = 8  # Arbitrary padded rows do not contribute.
        actual, support = statistics(scores, weights, torch.tensor([4, 2]), torch.ones(2, 4, dtype=torch.bool))
        for head in actual["all_rows"]:
            self.assertEqual(head["row_count"], 6)
            self.assertEqual(head["visible_pair_count"], 13)
            self.assertEqual(head["mean_support_share"], 1)
            self.assertEqual(head["exact_zero_pair_share"], 0)
        self.assertEqual(int(support[1, :, 2:].sum()), 0)
        single, _ = statistics(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1), torch.tensor([1]),
                               torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(single["all_rows"][0]["normalized_entropy"], 0)
        self.assertEqual(single["all_rows"][0]["self_only_row_share"], 1)
        self.assertEqual(single["nontrivial_rows"][0]["row_count"], 0)

    def test_recall_uses_latest_write_not_an_unrelated_equal_value(self):
        key, other, first, latest = (IDENTITY_BASE + offset for offset in (0, 1, 8, 9))
        tokens = [[BOS, key, first, other, latest, key, latest, latest, key, other]]
        targets = [[IGNORE] * 8 + [latest, latest]]
        queries = recall_queries(tokens, targets, pairs=3)
        self.assertEqual([query["answer_position"] for query in queries], [6, 4])
        weights = torch.zeros(1, 2, 10, 10)
        weights[0, :, 8, 6] = torch.tensor([0., .75])
        weights[0, :, 9, 4] = torch.tensor([.25, 0.])
        weights[0, :, 8, 4] = 1  # Same token, wrong key: excluded from the answer's weight.
        routes = answer_routes(weights, queries)
        self.assertEqual(routes["weights_per_head"], [[0., .75], [.25, 0.]])
        self.assertEqual(routes["maximum_weight"], [.75, .25])
        self.assertEqual(routes["in_any_support"], [True, True])
        weights[0, :, 8, 6] = 0
        self.assertEqual(answer_routes(weights, queries)["in_any_support"], [False, True])


class AttentionObserverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch._dynamo.reset()

    def test_new_block_keeps_existing_diagnostic_descriptions_compatible(self):
        self.assertEqual(describe(Diagnostics()), {"type": "Diagnostics", "every": 0,
                                                   "neighbors": False, "gradients": False})
        base = recall()
        require_continuation(describe(base), base)
        self.assertIsInstance(AttentionDiagnostics(), Diagnostics)
        for examples in (0, -1, .5, True):
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                AttentionDiagnostics(examples=examples)

    def test_observations_preserve_all_live_state_and_restore_modes_and_hooks_on_failure(self):
        for weights, projections in ((Softmax(), None), (Sparsemax(), None), (Softmax(fused=True), PerHeadQKV())):
            run = substitute(recall(), Softmax, weights)
            if projections is not None:
                run = swap(run, "model.block.attention.projections", projections)
            task = build_task(run.benchmark, run.seeds.data, torch.device("cpu"))
            model = build_model(run.model, task.vocab, run.seeds.model)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.step()
            model.train()
            model.blocks[0].attention.eval()  # Preserve mixed per-module modes, not only model.training.
            sampler = task.sampler(run.budget.batch, run.seeds.batch_seed)
            sampler.next()

            def snapshot():
                return summary({"model": model.state_dict(), "buffers": list(model.named_buffers()),
                                "gradients": [parameter.grad for parameter in model.parameters()],
                                "optimizer": optimizer.state_dict(), "sampler": sampler.state(),
                                "rng": torch.get_rng_state(), "modes": [module.training for module in model.modules()],
                                "hooks": [list(module._forward_pre_hooks) for module in model.modules()],
                                "inputs": {name: rows.rows for name, rows in task.splits.items()}})

            with self.subTest(weights=type(weights).__name__, projections=projections):
                before = snapshot()
                observer = AttentionObserver(AttentionDiagnostics(examples=8), task)
                first = observer.observe(model, task, 0)
                second = observer.observe(model, task, 10)
                self.assertEqual(snapshot(), before)
                self.assertEqual(first["rows_sha256"], second["rows_sha256"])
                self.assertEqual(len(first["queries"]), 16)
                self.assertEqual(len(first["layers"]), 2)
                for layer in second["layers"].values():
                    self.assertEqual([head["turnover_changed_pairs"] for head in layer["actual"]["all_rows"]], [0, 0])
                    self.assertEqual(len(layer["answer_routes"]["weights_per_head"]), 16)
                state = summary(observer.state_dict())
                with torch.no_grad():
                    model.blocks[0].attention.scores.log_alpha.add_(2)
                before = snapshot()
                forward = task.forward

                def failed(model, batch):
                    forward(model, batch)
                    raise RuntimeError("deliberate failed observation")

                with patch.object(task, "forward", failed):
                    with self.assertRaisesRegex(RuntimeError, "deliberate"):
                        observer.observe(model, task, 20)
                self.assertEqual(snapshot(), before)
                self.assertEqual(summary(observer.state_dict()), state)

    def test_temporary_hooks_do_not_invalidate_a_compiled_model(self):
        run = substitute(recall(), Softmax, Sparsemax())
        task = build_task(run.benchmark, run.seeds.data, torch.device("cpu"))
        model = build_model(run.model, task.vocab, run.seeds.model).eval()
        observer = AttentionObserver(AttentionDiagnostics(examples=8), task)
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        with torch.no_grad():
            tokens = observer.fixed.batch[:, 0]
            expected = compiled(tokens)
            for step in range(3):
                observer.observe(model, task, step)
                self.assertTrue(torch.equal(compiled(tokens), expected))
        self.assertEqual(len(graphs), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_observation_preserves_parameters_gradients_and_all_rng_states(self):
        for weights in (Softmax(fused=True), Sparsemax()):
            run = substitute(recall(), Softmax, weights)
            task = build_task(run.benchmark, run.seeds.data, torch.device("cuda"))
            model = build_model(run.model, task.vocab, run.seeds.model).cuda()
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.step()
            sampler = task.sampler(run.budget.batch, run.seeds.batch_seed)

            def snapshot():
                return summary({"model": model.state_dict(), "buffers": list(model.named_buffers()),
                                "gradients": [parameter.grad for parameter in model.parameters()],
                                "optimizer": optimizer.state_dict(), "sampler": sampler.state(),
                                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                                "modes": [module.training for module in model.modules()]})

            with self.subTest(weights=type(weights).__name__):
                before = snapshot()
                observer = AttentionObserver(AttentionDiagnostics(examples=8), task)
                observer.observe(model, task, 0)
                self.assertEqual(snapshot(), before)

    def test_fixed_batch_uses_the_selected_validation_split_and_modular_heldout(self):
        base = recall()
        selected = swap(swap(swap(base, "model.context", 23), "benchmark.ood", (23,)),
                        "benchmark.select", "length-23")
        runs = (selected,
                modular(gptmini(16, 1, 2), prime=11, updates=10, every=5))
        for run, split in zip(runs, ("validation/length-23", "heldout"), strict=True):
            run = swap(run, "model.context", run.benchmark.context)
            task = build_task(run.benchmark, run.seeds.data, torch.device("cpu"))
            model = build_model(run.model, task.vocab, run.seeds.model)
            observer = AttentionObserver(AttentionDiagnostics(examples=8), task)
            row = observer.observe(model, task, 0)
            self.assertEqual(row["split"], split)
            self.assertEqual(row["examples"], 8)
            self.assertEqual(observer.fixed.lengths.tolist(), [23 if split.startswith("validation/") else 6] * 8)

    def test_measured_compiled_runs_have_the_same_losses_and_checkpoint_state(self):
        for weights in (Softmax(), Sparsemax()):
            base = substitute(recall(), Softmax, weights)
            measured = swap(base, "diagnostics", AttentionDiagnostics(every=5, examples=8))
            with self.subTest(weights=type(weights).__name__), tempfile.TemporaryDirectory() as root:
                plain, sampled = Path(root) / "plain", Path(root) / "sampled"
                train(base, plain, CompiledStepper)
                train(measured, sampled, CompiledStepper)
                a, b = RunDirectory(plain), RunDirectory(sampled)
                self.assertEqual(untimed(a.records("history")), untimed(b.records("history")))
                state = b.checkpoint("cpu")
                self.assertIsNotNone(state.pop("attention"))
                self.assertEqual(summary(a.checkpoint("cpu")), summary(state))
                self.assertEqual(summary(a.result()), summary(b.result()))
                self.assertEqual(a.records("diagnostics"), b.records("diagnostics"))
                observations = b.records("attention")
                self.assertEqual([row["step"] for row in observations], [0, 5, 10, 15, 20])
                self.assertGreater(b.result()["diagnostic_seconds"], 0)

    def test_resumed_support_turnover_and_records_equal_an_uninterrupted_run(self):
        for weights in (Softmax(), Sparsemax()):
            run = swap(substitute(recall(), Softmax, weights), "diagnostics", AttentionDiagnostics(every=5, examples=8))

            def interrupt(row):
                if row["step"] == 15:
                    raise Interrupted

            with self.subTest(weights=type(weights).__name__), tempfile.TemporaryDirectory() as root:
                straight, resumed = Path(root) / "straight", Path(root) / "resumed"
                train(run, straight, CompiledStepper)
                with self.assertRaises(Interrupted):
                    train(run, resumed, CompiledStepper, interrupt)
                self.assertEqual(RunDirectory(resumed).checkpoint("cpu")["step"], 10)
                self.assertEqual(RunDirectory(resumed).records("attention")[-1]["step"], 15)
                train(run, resumed, CompiledStepper)
                self.assertEqual(records(resumed), records(straight))

    def test_diagnostic_neighbors_measure_attention_without_changing_the_control(self):
        base = swap(recall(), "diagnostics", Diagnostics(every=5, neighbors=True))
        measured = swap(base, "diagnostics", AttentionDiagnostics(every=5, neighbors=True, examples=8))
        with tempfile.TemporaryDirectory() as root:
            plain, sampled = Path(root) / "plain", Path(root) / "sampled"
            train(base, plain, CompiledStepper)
            train(measured, sampled, CompiledStepper)
            a, b = RunDirectory(plain), RunDirectory(sampled)
            self.assertEqual(untimed(a.records("history")), untimed(b.records("history")))
            self.assertEqual(untimed(a.records("probes")), untimed(b.records("probes")))
            observations = b.records("attention")
            expected = sorted([row["step"] for row in a.records("history") + a.records("probes")])
            self.assertEqual([row["step"] for row in observations], expected)
            self.assertEqual([row["step"] for row in observations if row["diagnostic_probe"]],
                             [row["step"] for row in a.records("probes")])


if __name__ == "__main__":
    unittest.main()
