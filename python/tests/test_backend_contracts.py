"""Backend certificates reject changed/nonfinite data and replay frozen evidence."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


REVIEW = Path(__file__).resolve().parents[2] / "review"
SPEC = importlib.util.spec_from_file_location("backend_contract_arithmetic", REVIEW / "backend_contract_arithmetic.py")
ARITHMETIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARITHMETIC)


class BackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((REVIEW / "backend_contract_check.json").read_text())

    def test_captured_premises_and_exact_decodings_replay(self):
        cases = self.report["cases"]
        self.assertEqual([(c["device"], c["dtype"], c["compile"]) for c in cases],
                         [("cpu", "torch.float32", "eager"), ("cpu", "torch.float64", "eager"),
                          ("cuda", "torch.float32", "max-autotune/fullgraph/static")])
        for case in cases:
            with self.subTest(device=case["device"], dtype=case["dtype"]):
                self.assertEqual(ARITHMETIC.check(case["outputs"]), case["checks"])
                self.assertEqual(case["sdpa_backend"], "MATH")
                self.assertTrue(case["observed_events"])

    def test_evidence_matches_capture_and_checker_sources(self):
        for name, expected in self.report["scripts"].items():
            self.assertEqual(hashlib.sha256((REVIEW / name).read_bytes()).hexdigest(), expected, name)
        pinned = json.loads((REVIEW / "sequence_model_source.json").read_text())
        self.assertEqual(self.report["source_commit"], pinned["repository_commit"])
        self.assertEqual(self.report["source_hashes"], pinned["files"])

    def test_lean_literals_match_the_frozen_binary_outputs(self):
        spec = importlib.util.spec_from_file_location("backend_contract_lean", REVIEW / "backend_contract_lean.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        for path, expected in exporter.render(self.report).items():
            self.assertEqual((REVIEW.parent / path).read_text(), expected, str(path))
            self.assertNotIn("native_decide", expected.split("namespace", 1)[1])

    def test_each_operation_rejects_a_corrupted_output(self):
        for name in ARITHMETIC.NAMES:
            with self.subTest(operation=name):
                outputs = copy.deepcopy(self.report["cases"][0]["outputs"])
                outputs[name][0] += .01
                with self.assertRaisesRegex(ValueError, f"{name}: rational certificate failed"):
                    ARITHMETIC.check(outputs)

    def test_nonfinite_values_never_become_real_traces(self):
        for value in (float("inf"), -float("inf"), float("nan"), True, "1.0"):
            with self.subTest(value=value):
                outputs = copy.deepcopy(self.report["cases"][0]["outputs"])
                outputs["rms"][0] = value
                with self.assertRaises(ValueError):
                    ARITHMETIC.check(outputs)

    def test_missing_extra_or_truncated_operations_are_rejected(self):
        for change in (lambda x: x.pop("cos"), lambda x: x.update(extra=[0]),
                       lambda x: x["sdpa"].pop()):
            outputs = copy.deepcopy(self.report["cases"][0]["outputs"])
            change(outputs)
            with self.assertRaises(ValueError):
                ARITHMETIC.check(outputs)

    def test_rotary_layout_and_causal_readout_are_not_interchangeable(self):
        outputs = copy.deepcopy(self.report["cases"][0]["outputs"])
        outputs["cos"].reverse()
        with self.assertRaisesRegex(ValueError, "cos"):
            ARITHMETIC.check(outputs)
        outputs = copy.deepcopy(self.report["cases"][0]["outputs"])
        outputs["sdpa"] = [1., 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, "sdpa"):
            ARITHMETIC.check(outputs)


if __name__ == "__main__":
    unittest.main()
