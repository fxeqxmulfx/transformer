"""Dense gradients preserve training and remain complete after checkpoint resume."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage
from experiments.synthetic_trainers.stability_integrity import validate_logs
from experiments.synthetic_trainers.stability_report import parse_rows, save_run, verify_archive
from experiments.synthetic_trainers.tests.test_modular_diagnostics import assert_checkpoint_equal, read_rows


class GradientTraceTests(unittest.TestCase):
    def config(self):
        return RunConfig(model="gptmini", optimizer="amsgradw", prime=7,
                         train_fraction=.5, width=8, heads=1, layers=1,
                         steps=12, eval_every=3, batch_size=8, device="cpu")

    def test_dense_trace_preserves_weights_moments_scores_and_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = train(self.config(), root / "plain", diagnostics=DiagnosticsConfig(3, True))
            traced = train(self.config(), root / "traced", diagnostics=DiagnosticsConfig(3, True, True))
            assert_checkpoint_equal(self, torch.load(root / "plain/checkpoint.pt", weights_only=True),
                                    torch.load(root / "traced/checkpoint.pt", weights_only=True))
            self.assertEqual([p[s] for p in plain["history"] for s in ("train", "heldout")],
                             [p[s] for p in traced["history"] for s in ("train", "heldout")])
            gradients = read_rows(root / "traced/gradients.jsonl")
            self.assertEqual([p["step"] for p in gradients], list(range(1, 13)))
            self.assertEqual({p["batch_size"] for p in gradients}, {5, 8})
            self.assertEqual(gradients[0]["learning_rate"], 0)
            validate_logs(traced, read_rows(root / "traced/diagnostics.jsonl"),
                          read_rows(root / "traced/probes.jsonl"), {"instrumentation": asdict(DiagnosticsConfig(3, True, True)),
                          "corpus": traced["plan"]["corpus"]}, gradients)

    def test_resume_discards_uncheckpointed_trace_without_losing_prior_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = DiagnosticsConfig(3, True, True)
            train(self.config(), root / "full", diagnostics=diagnostics)
            train(replace(self.config(), steps=6), root / "resumed", diagnostics=diagnostics)
            with (root / "resumed/gradients.jsonl").open("a") as stream:
                stream.write('{"step": 7, "uncheckpointed": true}\n')
            train(self.config(), root / "resumed", resume=True, diagnostics=diagnostics)
            assert_checkpoint_equal(self, torch.load(root / "full/checkpoint.pt", weights_only=True),
                                    torch.load(root / "resumed/checkpoint.pt", weights_only=True))
            self.assertEqual(read_rows(root / "full/gradients.jsonl"), read_rows(root / "resumed/gradients.jsonl"))
            path = root / "resumed/gradients.jsonl"
            rows = read_rows(path)
            path.write_text("".join(json.dumps(p) + "\n" for p in rows[1:]))
            with self.assertRaisesRegex(ValueError, "Checkpointed gradient trace"):
                train(replace(self.config(), steps=15), root / "resumed", resume=True, diagnostics=diagnostics)

    def test_complete_trace_archive_survives_without_training_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipes = [{"name": "trace-control", "config": asdict(self.config())}]
            manifest = freeze(root / "source", recipes, DiagnosticsConfig(3, True, True), PersistenceConfig())
            run_stage(root / "source", manifest)
            source = root / "source/trace-control"
            report = json.loads((source / "measurements.json").read_text())
            diagnostics = read_rows(source / "diagnostics.jsonl")
            probes = read_rows(source / "probes.jsonl")
            gradients = read_rows(source / "gradients.jsonl")
            for failure in ("missing", "duplicate", "batch", "rate", "norm", "disagreement"):
                bad = deepcopy(gradients)
                if failure == "missing":
                    bad.pop()
                elif failure == "duplicate":
                    bad[-1] = bad[-2]
                elif failure == "batch":
                    bad[0]["batch_size"] += 1
                elif failure == "rate":
                    bad[0]["learning_rate"] = 1
                elif failure == "norm":
                    bad[0]["gradient_l2"] = float("nan")
                else:
                    bad[0]["gradient_l2"] += 1
                with self.subTest(failure=failure), self.assertRaises(ValueError):
                    validate_logs(report, diagnostics, probes, manifest, bad)
            summary = save_run(root / "source", "trace-control", root / "archive", render=False)
            self.assertEqual(summary["gradient_trace"]["observations"], 12)
            self.assertEqual(parse_rows((root / "archive/gradients.jsonl").read_bytes()), gradients)
            self.assertEqual(len((root / "archive/gradient-metrics.csv").read_text().splitlines()), 13)
            shutil.rmtree(root / "source")
            self.assertEqual(verify_archive(root / "archive"), summary)
            result = subprocess.run([sys.executable, "-c",
                "import sys; from experiments.synthetic_trainers.stability_report import verify_archive; "
                "verify_archive(sys.argv[1]); assert 'torch' not in sys.modules", str(root / "archive")],
                capture_output=True, text=True, check=True)
            self.assertEqual(result.returncode, 0)
