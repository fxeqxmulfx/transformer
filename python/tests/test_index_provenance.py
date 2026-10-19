"""Module-system headers must not erase the generated index's paper attribution."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/index.py"
SPEC = importlib.util.spec_from_file_location("project_index", SCRIPT)
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


class IndexProvenanceTests(unittest.TestCase):
    def citation(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example.lean"
            path.write_text(source)
            return INDEX.arxiv_id(path)

    def test_long_license_and_module_headers_preserve_the_source_citation(self):
        header = "/-\n" + "License and attribution.\n" * 14 + "-/\n"
        body = '/-! Source: arXiv:2510.22026v2, Appendix D.1. -/\ndef energy : ℕ := 1\n'
        self.assertEqual(self.citation(header + body), "2510.22026")
        self.assertEqual(self.citation(header + "module\n\npublic import Mathlib\n\n" + body),
                         "2510.22026")

    def test_a_citation_in_a_declaration_body_does_not_attribute_the_module(self):
        self.assertIsNone(self.citation('module\n\ndef exampleText := "arXiv:2510.22026v2"\n'))


if __name__ == "__main__":
    unittest.main()
