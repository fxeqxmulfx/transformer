"""Regression references must run from local files in a checkout without Git."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import reference_fixtures


ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceSourceTests(unittest.TestCase):
    def test_all_references_extract_in_an_independent_copy_without_git(self):
        sources = script("reference_sources")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / sources.BUNDLES, root / sources.BUNDLES)
            shutil.copytree(ROOT / sources.FROZEN, root / sources.FROZEN)
            shutil.copy2(ROOT / "python/tests/reference_fixtures.py", root / "python/tests/reference_fixtures.py")
            with patch.object(subprocess, "check_output", side_effect=AssertionError("Git is unavailable")), \
                    patch.object(reference_fixtures, "REFERENCES", root / "python/tests/references"), \
                    contextlib.redirect_stdout(io.StringIO()):
                sources.check(root)
                for kind, revision in reference_fixtures.REVISIONS.items():
                    destination = root / kind
                    destination.mkdir()
                    reference_fixtures.export_sources(revision, destination)
                    self.assertTrue(any(destination.rglob("*.py")), kind)
                    for path in destination.rglob("*"):
                        self.assertTrue(path.resolve().is_relative_to(destination), path)
                        self.assertTrue(path.exists(), path)
            self.assertFalse((root / ".git").exists())

    def test_missing_or_changed_reference_fails_without_fallback(self):
        revision = reference_fixtures.REVISIONS["modular"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references"
            shutil.copytree(reference_fixtures.REFERENCES / "sources", references / "sources")
            archive = references / "sources" / f"{revision}.tar.gz"
            destination = root / "output"
            destination.mkdir()
            with patch.object(reference_fixtures, "REFERENCES", references), \
                    patch.object(subprocess, "check_output", side_effect=AssertionError("Git is unavailable")):
                archive.write_bytes(archive.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "source has changed"):
                    reference_fixtures.export_sources(revision, destination)
                archive.unlink()
                with self.assertRaises(FileNotFoundError):
                    reference_fixtures.export_sources(revision, destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_external_archive_links_are_rejected(self):
        sources = script("reference_sources")
        for target in ("/another-checkout/model.py", "../../another-checkout/model.py"):
            with self.subTest(target=target):
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w") as archive:
                    link = tarfile.TarInfo("model.py")
                    link.type = tarfile.SYMTYPE
                    link.linkname = target
                    archive.addfile(link)
                with self.assertRaisesRegex(ValueError, "external|External"):
                    sources.archive_members(data.getvalue())

    def test_measured_source_identifiers_resolve_only_to_local_snapshots(self):
        benchmark = script("optimizer_benchmark_data")
        sources = script("reference_sources")
        manifest = json.loads((ROOT / sources.BUNDLES / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / sources.FROZEN
            shutil.copytree(ROOT / sources.FROZEN, frozen)
            with patch.object(benchmark, "ROOT", root), patch.object(benchmark, "FROZEN", frozen), \
                    patch.object(subprocess, "run", side_effect=AssertionError("Git is unavailable")):
                for name, digest in manifest["benchmark"]["files"].items():
                    identifier = f"/nonexistent-checkout/experiments/optimizer_benchmark/{name}"
                    self.assertEqual(hashlib.sha256(benchmark.measured_source(identifier)).hexdigest(), digest)
                for identifier in ("/another-checkout/model.py", "../model.py",
                                   "/old/experiments/optimizer_benchmark/../../model.py"):
                    with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                        benchmark.measured_source(identifier)


if __name__ == "__main__":
    unittest.main()
