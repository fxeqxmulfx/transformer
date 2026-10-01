"""Enforce inward dependencies and import the inner layers without PyTorch."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


class ArchitectureTests(unittest.TestCase):
    def test_domain_and_application_dependencies_point_inward(self):
        for layer in ("domain", "application"):
            root = Path(importlib.util.find_spec(f"gpt_mini.{layer}").submodule_search_locations[0])
            for path in root.rglob("*.py"):
                with self.subTest(file=path.name):
                    tree = ast.parse(path.read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                            for name in names:
                                self.assertNotIn(name.split('.')[0], {"torch", "numpy", "pathlib", "subprocess", "os", "json"})
                                self.assertNotIn("infrastructure", name)
                                self.assertNotIn("interfaces", name)
                                if layer == "domain":
                                    self.assertNotIn("application", name)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            self.assertNotEqual(node.func.id, "open")

    def test_inner_layers_import_without_loading_tensor_libraries(self):
        code = "import sys; from gpt_mini.application.benchmark import RunBenchmark; from gpt_mini.domain.stopping import EarlyStopper; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_cli_help_and_optimizer_selection_do_not_require_cuda(self):
        result = subprocess.run([sys.executable, "-B", "-m", "gpt_mini", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("27 optimizers", result.stdout)
        self.assertIn("--max-steps", result.stdout)
        self.assertIn("--all", result.stdout)

        from gpt_mini.domain.benchmark import Request, plan_jobs
        from gpt_mini.interfaces.cli import parser

        rates = {attention: {"amsgradw": .0003, "adamw": .001}
                 for attention in ("softmax", "sparsemax")}
        self.assertEqual({job.method for job in plan_jobs(Request(), rates)}, {"amsgradw"})
        for arguments, methods in (([], {"amsgradw"}), (["--all"], {"amsgradw", "adamw"}),
                                   (["--only", "adamw"], {"adamw"})):
            with self.subTest(arguments=arguments):
                args = parser().parse_args(arguments)
                jobs = plan_jobs(Request(methods=tuple(args.only)), rates)
                expected = {(attention, method, seed) for attention in rates
                            for method in methods for seed in (0, 1, 2)}
                self.assertEqual({(job.attention, job.method, job.seed) for job in jobs}, expected)
                self.assertEqual(len(jobs), len(expected))
