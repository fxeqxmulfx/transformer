"""Local trace certificates reject errors hidden by the old uniform allowance."""

import copy
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
    sys.modules[name] = module
    return module


load("backend_contract_arithmetic")
COARSE = load("model_trace_arithmetic")
load("model_trace_lean")
REFINED = load("model_trace_refined")


class RefinedModelTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = json.loads((REVIEW / "model_trace_capture.json").read_text())

    def test_all_local_checks_and_generated_proofs_replay(self):
        certificate = REFINED.check_refined(self.capture)
        self.assertEqual(len(certificate["maximum_rational_residuals"]), 52)
        generated = REFINED.render(self.capture, certificate)
        self.assertEqual(len(generated), 7)
        for path, expected in generated.items():
            self.assertEqual((REVIEW.parent / path).read_text(), expected, str(path))

    def test_small_pre_norm_error_passes_old_allowance_but_is_rejected(self):
        changed = copy.deepcopy(self.capture)
        changed["trace"]["pre"]["values"][0] += 1e-7
        COARSE.check(changed)
        with self.assertRaisesRegex(ValueError, "pre.0: refined"):
            REFINED.check_refined(changed)

    def test_small_q_projection_error_requires_separate_component_check(self):
        changed = copy.deepcopy(self.capture)
        changed["parameters"]["blocks.0.sa.c_attn.weight"]["values"][0] += 1e-9
        COARSE.check(changed)
        with self.assertRaisesRegex(ValueError, "component: refined"):
            REFINED.check_refined(changed)

    def test_late_addition_cannot_reuse_an_early_exact_residual(self):
        changed = copy.deepcopy(self.capture)
        changed["trace"]["output"]["values"][0] += 2e-7
        COARSE.check(changed)
        with self.assertRaisesRegex(ValueError, "output.0: refined"):
            REFINED.check_refined(changed)


if __name__ == "__main__":
    unittest.main()
