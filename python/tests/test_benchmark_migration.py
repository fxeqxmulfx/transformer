"""Migration contracts for numerical rules, evidence, data, and installation."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from gpt_mini.infrastructure.benchmark.paths import DATA_FILE, MIGRATION, PACKAGE_ROOT, WORK_ROOT, archive_path, artifact_path
from gpt_mini.infrastructure.benchmark.provenance import audit_archive, source_hashes, source_snapshot_matches


class MigrationTests(unittest.TestCase):
    def test_dataset_bytes_and_original_split(self):
        from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData
        self.assertEqual(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(), MIGRATION["dataset_sha256"])
        data = TextData.load(DATA_FILE)
        self.assertEqual(len(data.characters), 65)
        original = json.loads(archive_path("experiments/full_compile_benchmark/results/rtx3050_all/metadata.json").read_text())["protocol"]
        self.assertEqual(data.sha256, original["data_sha256"])
        self.assertEqual(list(data.boundaries), original["data_boundaries"])

    def test_all_historical_sources_and_reports_keep_their_hashes(self):
        self.assertGreater(audit_archive(), 194)

    def test_all_27_recipes_and_selected_rates_are_available(self):
        from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import METHODS, selected_rates
        from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.protocol import METHODS as extensions
        self.assertEqual(len({m.name for m in METHODS} | set(extensions)), 27)
        rates = selected_rates()
        for attention in ("softmax", "sparsemax"):
            self.assertEqual(set(rates[attention]), {m.name for m in METHODS})
        for name in MIGRATION["benchmark_packages"]:
            importlib.import_module(f"gpt_mini.infrastructure.benchmark.{name}.runner")

    def test_frozen_baselines_remain_auditable(self):
        from gpt_mini.infrastructure.benchmark.magma_benchmark.protocol import baseline
        from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.protocol import baseline as all24
        self.assertEqual(len(baseline()[0]), 216)
        self.assertEqual(len(all24()[0]), 144)

    def test_active_and_historical_source_fingerprints_are_distinct(self):
        active = source_hashes()
        self.assertTrue(source_snapshot_matches(active))
        self.assertTrue(source_snapshot_matches(MIGRATION["measured_sources"]))
        changed = dict(active)
        changed[next(iter(changed))] = "0" * 64
        self.assertFalse(source_snapshot_matches(changed))
        self.assertNotEqual(active, MIGRATION["measured_sources"])

    def test_legacy_checkpoint_paths_resolve_to_the_new_run_directory(self):
        old = "experiments/runs/full_compile_benchmark/rtx3050_all/model.pt"
        expected = WORK_ROOT / "runs/full_compile_benchmark/rtx3050_all/model.pt"
        self.assertEqual(artifact_path(old), expected)
        self.assertEqual(artifact_path(Path(MIGRATION["original_workspace"]) / old), expected)

    def test_numerical_kernels_keep_the_original_ast(self):
        names = {
            "optimizer_benchmark": ("common", "coordinate", "matrix", "roots", "fisher", "attention", "data"),
            "magma_benchmark": ("optimizer", "registry"),
            "patience_benchmark": ("stopping",),
            "full_compile_benchmark": ("optimizer", "model", "step"),
            "amsgrad_extensions_benchmark": ("optimizer", "step"),
        }

        class Normalize(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

            def visit_Import(self, node):
                return None

        for package, modules in names.items():
            for module in modules:
                with self.subTest(package=package, module=module):
                    old = archive_path(f"experiments/{package}/{module}.py")
                    new = (PACKAGE_ROOT.parents[1] / "domain/stopping.py" if package == "patience_benchmark"
                           else PACKAGE_ROOT / package / f"{module}.py")
                    self.assertEqual(ast.dump(Normalize().visit(ast.parse(old.read_text()))),
                                     ast.dump(Normalize().visit(ast.parse(new.read_text()))))
