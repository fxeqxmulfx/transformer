"""Load the packaged CPU/CUDA contracts independently of the current directory."""

import importlib
import pkgutil
import unittest

from .paths import MIGRATION


def test_suite(names=None):
    names = MIGRATION["benchmark_packages"] if names is None else names
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in names:
        package = importlib.import_module(f"gpt_mini.infrastructure.benchmark.tests.{name}")
        for module in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if module.name.startswith("test_"):
                suite.addTests(loader.loadTestsFromModule(importlib.import_module(f"{package.__name__}.{module.name}")))
    return suite
