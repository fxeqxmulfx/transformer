"""Full trace certificates: coverage, provenance, rejection and exact Lean replay."""

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


REVIEW = Path(__file__).resolve().parents[2] / "review"


def load(name):
    spec = importlib.util.spec_from_file_location(name, REVIEW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("backend_contract_arithmetic", load("backend_contract_arithmetic"))
CHECKER = load("model_trace_arithmetic")
EXPORTER = load("model_trace_lean")


class ModelTraceCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = json.loads((REVIEW / "model_trace_capture.json").read_text())
        cls.certificates = json.loads((REVIEW / "model_trace_certificates.json").read_text())

    def test_all_52_site_checks_replay(self):
        result = CHECKER.check(self.capture)
        self.assertEqual(len(result["maximum_rational_residuals"]), 52)
        for key, value in result.items():
            self.assertEqual(value, self.certificates[key])
        self.assertEqual(self.capture["compile"], "max-autotune/fullgraph/static")
        self.assertEqual(self.capture["sdpa_backend"], "MATH")
        self.assertEqual(self.capture["instrumentation_max_logit_difference"], 0)
        self.assertEqual(self.capture["uninstrumented_logits"], self.capture["trace"]["logits"])

    def test_provenance_hashes_are_current(self):
        for name, expected in (("model_trace_capture.py", self.capture["script_sha256"]),
                               ("model_trace_capture.json", self.certificates["capture_sha256"]),
                               ("model_trace_arithmetic.py", self.certificates["checker_sha256"])):
            self.assertEqual(hashlib.sha256((REVIEW / name).read_bytes()).hexdigest(), expected)
        source = json.loads((REVIEW / "sequence_model_source.json").read_text())
        self.assertEqual(self.capture["source_hashes"], source["files"])
        self.assertEqual(self.capture["source_commit"], source["repository_commit"])

    def test_exported_lean_contains_exact_capture_and_witnesses(self):
        for path, expected in EXPORTER.render(self.capture, self.certificates).items():
            self.assertEqual((REVIEW.parent / path).read_text(), expected, str(path))

    def test_every_trace_boundary_rejects_corruption(self):
        for name in self.capture["trace"]:
            with self.subTest(site=name):
                changed = copy.deepcopy(self.capture)
                changed["trace"][name]["values"][0] += .01
                with self.assertRaises(ValueError):
                    CHECKER.check(changed)

    def test_parameters_and_rotary_tables_are_checked(self):
        for group in ("parameters", "tables"):
            for name in self.capture[group]:
                with self.subTest(group=group, site=name):
                    changed = copy.deepcopy(self.capture)
                    changed[group][name]["values"][0] += .01
                    with self.assertRaises(ValueError):
                        CHECKER.check(changed)

    def test_nonfinite_and_malformed_traces_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "0"):
            changed = copy.deepcopy(self.capture)
            changed["trace"]["initial"]["values"][0] = value
            with self.assertRaises(ValueError):
                CHECKER.check(changed)
        for mutation in (lambda r: r["trace"].pop("mid"),
                         lambda r: r["trace"].update(extra=r["trace"]["mid"]),
                         lambda r: r["trace"]["scores"].update(shape=[2, 2]),
                         lambda r: r["trace"]["gate"]["values"].pop(),
                         lambda r: r.update(tokens=[1, 0]),
                         lambda r: r.update(rms_epsilon="1/1000")):
            changed = copy.deepcopy(self.capture)
            mutation(changed)
            with self.assertRaises(ValueError):
                CHECKER.check(changed)

    def test_branches_decisions_and_unavoidable_residual_error(self):
        trace = self.capture["trace"]
        for name in ("attn_output", "ffn_output"):
            self.assertGreater(sum(abs(x) for x in trace[name]["values"][4:]), 1)
        logits = trace["logits"]["values"]
        self.assertEqual([max(range(3), key=lambda v: logits[3*i+v]) for i in range(2)], [0, 2])
        error = abs(Fraction(trace["output"]["values"][6]) -
                    Fraction(trace["mid"]["values"][6]) - Fraction(trace["ffn_output"]["values"][6]))
        self.assertEqual(error, Fraction(1, 2**24))


if __name__ == "__main__":
    unittest.main()
