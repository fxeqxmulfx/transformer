"""Exact selected controls, sparsemax continuation and portable paired failures."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import architecture_report as report
from experiments.synthetic_trainers import architecture_training as architecture
from experiments.synthetic_trainers import attention_training, scheduled_training, stability_report
from experiments.synthetic_trainers.attention_layout import current_sources
from experiments.synthetic_trainers.paper_reproduction import grokking
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_analysis import diagnostic_metrics
from experiments.synthetic_trainers.stability_protocol import environment, paper_fingerprints
from experiments.synthetic_trainers.tagged_confirmation_report import verified_benchmark
from experiments.synthetic_trainers.tests import test_scheduled_training as selected_tests


class ArchitectureTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **fields):
        return architecture.ArchitectureRunConfig(**(dict(model="gptmini", optimizer="adamw", prime=7,
            train_fraction=.5, steps=40, batch_size=4, eval_every=5, learning_rate=.0003,
            weight_decay=.1, anneal_start=20, anneal_end=30, device="cpu") | fields))

    def equal_runs(self, left, right):
        helper = selected_tests.ScheduledTrainingTests()
        helper.checkpoints(torch.load(left / "checkpoint.pt", weights_only=True),
                           torch.load(right / "checkpoint.pt", weights_only=True))
        helper.history(json.loads((left / "measurements.json").read_text()),
                       json.loads((right / "measurements.json").read_text()))
        for name in ("gradients.jsonl", "diagnostics.jsonl"):
            self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())

    def test_both_softmax_controls_and_constant_sparsemax_retain_selected_native_trajectories(self):
        diagnostics = DiagnosticsConfig(5, True, True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for schedule in ("constant", "cosine_tail"):
                cfg = self.config(learning_rate_schedule=schedule)
                selected = scheduled_training.ScheduledRunConfig(**{k: v for k, v in asdict(cfg).items()
                                                                    if k != "attention_normalization"})
                parent, control = root / f"parent-{schedule}", root / f"control-{schedule}"
                scheduled_training.train(selected, parent, diagnostics=diagnostics)
                architecture.train(cfg, control, diagnostics=diagnostics)
                self.equal_runs(parent, control)
                measured = json.loads((control / "measurements.json").read_text())
                self.assertEqual(measured["plan"]["source_hashes"], architecture.training_sources())
                self.assertEqual(len(architecture.training_sources()), 19)
            cfg = self.config(attention_normalization="sparsemax")
            remove = {"learning_rate_schedule", "anneal_start", "anneal_end", "final_rate_factor"}
            selected = attention_training.AttentionRunConfig(**{k: v for k, v in asdict(cfg).items() if k not in remove})
            parent, wrapped = root / "parent-sparsemax", root / "wrapped-sparsemax"
            attention_training.train(selected, parent, diagnostics=diagnostics)
            architecture.train(cfg, wrapped, diagnostics=diagnostics)
            self.equal_runs(parent, wrapped)

    def test_sparsemax_annealing_resume_keeps_buffers_sampling_logs_and_rejects_tag_changes(self):
        cfg = self.config(attention_normalization="sparsemax", learning_rate_schedule="cosine_tail")
        diagnostics = DiagnosticsConfig(5, True, True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fresh, continued = root / "fresh", root / "continued"
            architecture.train(cfg, fresh, diagnostics=diagnostics)
            architecture.train(replace(cfg, steps=30), continued, diagnostics=diagnostics)
            prefix = {name: (continued / name).read_bytes() for name in ("gradients.jsonl", "diagnostics.jsonl")}
            before = {p.name: p.read_bytes() for p in continued.iterdir() if p.is_file()}
            with patch.object(grokking.torch, "load", side_effect=AssertionError("reject before checkpoint load")):
                for changed in (replace(cfg, attention_normalization="softmax"),
                                replace(cfg, learning_rate_schedule="constant"), replace(cfg, final_rate_factor=.2)):
                    with self.assertRaises(ValueError):
                        architecture.train(changed, continued, resume=True, diagnostics=diagnostics)
            self.assertEqual(before, {p.name: p.read_bytes() for p in continued.iterdir() if p.is_file()})
            architecture.train(cfg, continued, resume=True, diagnostics=diagnostics)
            self.equal_runs(fresh, continued)
            for name, blob in prefix.items():
                self.assertTrue((continued / name).read_bytes().startswith(blob))

    def test_scientific_initial_tensors_and_rng_match_and_factory_restores_after_interruption(self):
        cfg = self.config(prime=193, train_fraction=.25, steps=300000, batch_size=512,
                          eval_every=250, anneal_start=150000, anneal_end=250000)
        vocabulary = len(make_corpus(193, .25, 0).tokens)
        original = grokking.make_model(cfg, vocabulary)
        state, rng = original.state_dict(), torch.get_rng_state()
        for norm in ("softmax", "sparsemax"):
            model = architecture.make_model(replace(cfg, attention_normalization=norm), vocabulary)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 436104)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(state.keys(), model.state_dict().keys())
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
        factory, rate, sources = grokking.make_model, grokking.learning_rate, grokking.source_hashes
        with tempfile.TemporaryDirectory() as tmp, patch.object(grokking, "evaluate", side_effect=RuntimeError("intentional")):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                architecture.train(self.config(), tmp)
        self.assertIs(grokking.make_model, factory); self.assertIs(grokking.learning_rate, rate)
        self.assertIs(grokking.source_hashes, sources)
        for fields in ({"attention_normalization": "softplus"}, {"steps": 300001}, {"steps": 40.5},
                       {"optimizer": "amsgradw"}, {"learning_rate_schedule": "adaptive"}):
            with self.assertRaises(ValueError):
                self.config(**fields)
        with self.assertRaises(TypeError):
            plain = scheduled_training.ScheduledRunConfig(**{k: v for k, v in asdict(self.config()).items()
                                                             if k != "attention_normalization"})
            architecture.train(plain, "unused")

    def test_complete_real_pairs_verify_without_torch_retain_failures_and_reject_rehashed_rates(self):
        diagnostics = DiagnosticsConfig(5, True, True)
        criterion = PersistenceConfig(plateau_steps=5, plateau_observations=2, confirmation_observations=2, tail_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stage = root / "stage"; stage.mkdir()
            recipes = [{"name": f"{schedule}-{norm}", "config": asdict(self.config(
                       learning_rate_schedule=schedule, attention_normalization=norm))}
                       for schedule in ("constant", "cosine_tail") for norm in ("softmax", "sparsemax")]
            manifest = {"recipes": recipes, "planned_runs": 4, "scientific_run": False,
                "stage": "CPU_architecture_schedule_fixture; no_learning_result", "criterion": asdict(criterion),
                "instrumentation": asdict(diagnostics), "training_source_hashes": architecture.training_sources(),
                "source_hashes": current_sources() | architecture.training_sources(), "papers": paper_fingerprints(),
                "environment": environment("cpu"), "corpus": make_corpus(7, .5, 0).summary()}
            stability_report.write_json(stage / "plan.json", manifest)
            summaries = {}
            for recipe in recipes:
                name = recipe["name"]
                architecture.train(architecture.ArchitectureRunConfig(**recipe["config"]), stage / name, diagnostics=diagnostics)
                output = root / name
                report.save_run(stage, name, output, render=False)
                summaries[name] = report.verify_archive(output)
            for schedule in ("constant", "cosine_tail"):
                result = report.paired_normalizer_outcome(summaries[f"{schedule}-softmax"], summaries[f"{schedule}-sparsemax"])
                self.assertEqual(result["eligible_pair_support"], 0)
                self.assertIsNone(result["control_to_candidate_training_time_ratio"])
                self.assertFalse(result["benchmark_gate_opened"])
                changed = dict(summaries[f"{schedule}-sparsemax"])
                changed["config"] = changed["config"] | {"anneal_start": 19}
                with self.assertRaises(ValueError):
                    report.paired_normalizer_outcome(summaries[f"{schedule}-softmax"], changed)
            shutil.rmtree(stage)
            code = ("from experiments.synthetic_trainers.architecture_report import verify_archive; import sys,json; "
                    "r=verify_archive(sys.argv[1]); print(json.dumps({'complete':r['complete_run'],'Torch_imported':'torch' in sys.modules}))")
            actual = json.loads(subprocess.check_output(["python3", "-c", code, str(root / "cosine_tail-sparsemax")], text=True))
            self.assertEqual(actual, {"complete": True, "Torch_imported": False})
            archive = root / "cosine_tail-sparsemax"
            rows = [json.loads(line) for line in (archive / "gradients.jsonl").read_text().splitlines()]
            rows[25]["learning_rate"] = .0003
            (archive / "gradients.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            stability_report.csv_rows(archive / "gradient-metrics.csv", rows)
            rows = [json.loads(line) for line in (archive / "diagnostics.jsonl").read_text().splitlines()]
            next(row for row in rows if row["step"] == 26)["learning_rate"] = .0003
            (archive / "diagnostics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            stability_report.csv_rows(archive / "diagnostic-metrics.csv", [diagnostic_metrics(row) for row in rows])
            stored = json.loads((archive / "artifact-hashes.json").read_text())
            stored["files"] = stability_report.file_hashes(archive)
            stability_report.write_json(archive / "artifact-hashes.json", stored)
            with self.assertRaisesRegex(ValueError, "scheduled learning rate"):
                report.verify_archive(archive)
        protocol = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
        for mode in ("normalizer", "schedule"):
            with self.assertRaises(ValueError):
                verified_benchmark(protocol / "tagged-confirmation-preparation" / mode)
