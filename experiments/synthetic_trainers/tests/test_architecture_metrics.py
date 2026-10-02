"""Failed pairs cannot certify speed, and persistent candidates need no plateau."""

from copy import deepcopy
from dataclasses import asdict, replace
import tempfile
from pathlib import Path
import unittest

from experiments.synthetic_trainers.architecture_metrics import (
    paired_outcome, timing_outcome, verified_benchmark,
)
from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.tests.test_confirmation import calibration_fixture
from experiments.synthetic_trainers.tests.test_persistence import measured
from experiments.synthetic_trainers.confirmation_protocol import freeze
from experiments.synthetic_trainers.confirmation_report import assemble
from experiments.synthetic_trainers.stability_confirmation import run_confirmation
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.stability import freeze as freeze_pair, run_stage
from experiments.synthetic_trainers.stability_report import save_run, verify_archive


def outcome(scores, *, candidate=False):
    report = measured(scores)
    criterion = PersistenceConfig(confirmation_observations=3, tail_steps=1000)
    config = {**report["plan"]["config"], "optimizer": "adamw", "prime": 7,
              "seed": 4, "data_seed": 2, "width": 16 if candidate else 8}
    final = {**report["final"], "epochs_seen": 100.0,
             "heldout": {**report["final"]["heldout"], "answer_loss": .02, "EOS_loss": .01}}
    return {"name": "candidate" if candidate else "control", "complete_run": True,
            "config": config, "assessment": assess(report, criterion), "final": final,
            "parameters": 200 if candidate else 100, "training_seconds": 15.0,
            "wall_seconds": 30.0, "diagnostic_seconds": .2,
            "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}


class ArchitectureMetricsTests(unittest.TestCase):
    def control(self):
        return outcome([0, *([.01] * 5), *([1] * 7)])

    def test_persistent_candidate_without_plateau_has_eligible_timing(self):
        candidate = outcome([0, *([1] * 12)], candidate=True)
        result = paired_outcome(self.control(), candidate, changed_fields=("width",))
        self.assertEqual(result["eligible_pair_support"], 1)
        self.assertFalse(result["candidate"]["stable_grokking"])
        self.assertEqual(result["candidate"]["phase_label"],
                         "persistent_generalization_without_required_memorization_plateau")
        self.assertEqual(result["control_to_candidate_training_time_ratio"], 8 / 3)
        self.assertEqual(result["control_to_candidate_wall_time_ratio"], 8 / 3)

    def test_late_failure_retains_observed_timing_and_quality_without_speed_ratio(self):
        candidate = outcome([0, *([.01] * 5), *([1] * 5), .1, 1], candidate=True)
        result = paired_outcome(self.control(), candidate, changed_fields=("width",))
        self.assertEqual(result["eligible_pair_support"], 0)
        self.assertIsNone(result["control_to_candidate_training_time_ratio"])
        self.assertEqual(result["candidate"]["observed_target_training_seconds"], 8)
        self.assertIsNone(result["candidate"]["eligible_target_training_seconds"])
        self.assertEqual(result["candidate"]["final_heldout_accuracy"], 1)
        self.assertEqual(result["candidate"]["tail_failures"], 1)

    def test_rapid_control_and_failed_control_do_not_open_timing_gate(self):
        candidate = outcome([0, *([1] * 12)], candidate=True)
        control = outcome([0, *([1] * 12)])
        self.assertEqual(paired_outcome(control, candidate, changed_fields=("width",))["eligible_pair_support"], 0)
        control = outcome([0, *([.01] * 5), *([1] * 5), .1, 1])
        self.assertEqual(paired_outcome(control, candidate, changed_fields=("width",))["eligible_pair_support"], 0)

    def test_missing_event_incomplete_or_invalid_costs_cannot_certify_timing(self):
        for case in ("event", "complete", "history"):
            candidate = outcome([0, *([1] * 12)], candidate=True)
            if case == "event":
                candidate["assessment"]["long_confirmation"] = None
            elif case == "complete":
                candidate["complete_run"] = False
            else:
                candidate["assessment"]["complete_canonical_history"] = False
            with self.subTest(case=case):
                result = timing_outcome(candidate)
                self.assertEqual(result["eligible_timing_support"], 0)
                self.assertIsNone(result["eligible_target_training_seconds"])
        for value in (None, 0, float("nan"), float("inf")):
            candidate = outcome([0, *([1] * 12)], candidate=True)
            candidate["assessment"]["long_confirmation"]["training_seconds"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite costs"):
                timing_outcome(candidate)

    def test_undeclared_task_optimizer_seed_and_criterion_changes_are_rejected(self):
        candidate = outcome([0, *([1] * 12)], candidate=True)
        with self.assertRaisesRegex(ValueError, "declared intervention"):
            paired_outcome(self.control(), candidate, changed_fields=("layers",))
        for key, value in (("optimizer", "amsgradw"), ("prime", 193), ("seed", 5),
                           ("data_seed", 3), ("steps", 150000), ("learning_rate", .001)):
            bad = deepcopy(candidate)
            bad["config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preserve AdamW"):
                paired_outcome(self.control(), bad, changed_fields=("width", key))
        bad = deepcopy(candidate)
        bad["assessment"]["criterion"]["target"] = .9
        with self.assertRaisesRegex(ValueError, "criteria differ"):
            paired_outcome(self.control(), bad, changed_fields=("width",))

    def test_real_complete_negative_adamw_confirmation_cannot_select_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Hypothetical passing calibration is only an orchestration fixture.
            calibration = calibration_fixture(root, hypothetical_passing=True, optimizer="adamw")
            plan = freeze(root / "stage", calibration, "short-lr001", render=False)
            run_confirmation(root / "stage", plan)
            assemble(root / "stage", root / "portable", render=False)
            with self.assertRaisesRegex(ValueError, "every frozen benchmark case"):
                verified_benchmark(root / "portable")

    def test_real_complete_paired_archives_keep_quality_costs_and_failed_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = RunConfig(model="gptmini", optimizer="adamw", prime=7, train_fraction=.5,
                width=8, heads=1, layers=1, steps=6, eval_every=2, batch_size=8, device="cpu")
            recipes = [{"name": "control", "config": asdict(base), "changed_mechanism": "unchanged_control"},
                       {"name": "candidate", "config": asdict(replace(base, width=16)), "changed_mechanism": "width"}]
            plan = freeze_pair(root / "source", recipes, DiagnosticsConfig(2, True, True), PersistenceConfig())
            run_stage(root / "source", plan)
            for row in recipes:
                save_run(root / "source", row["name"], root / "archives" / row["name"], render=False)
            control, candidate = (verify_archive(root / "archives" / name) for name in ("control", "candidate"))
            result = paired_outcome(control, candidate, changed_fields=("width",))
            self.assertEqual(result["eligible_pair_support"], 0)
            self.assertIsNone(result["control_to_candidate_training_time_ratio"])
            self.assertGreater(result["candidate"]["parameters"], result["control"]["parameters"])
            for name, summary in (("control", control), ("candidate", candidate)):
                self.assertEqual(result[name]["final_heldout_accuracy"], summary["final"]["heldout"]["accuracy"])
                self.assertEqual(result[name]["training_seconds"], summary["training_seconds"])
                self.assertEqual(result[name]["epochs_seen"], summary["final"]["epochs_seen"])
