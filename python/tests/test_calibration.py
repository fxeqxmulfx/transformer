"""The learning rates the convex MQAR comparison selected, chosen again over its experiment.

Golden records: `fixtures/legacy_rates.json`, what
`validation_milestones.build_report` of `experiments/convex_mqar` selected
under each policy from the archived summaries of its 24 runs, and what it
read of each. The runs are matched to the labels of
`experiments/mqar_sparsemax`, whose epochs set the observed updates. The
comparison ranked a best observation by accuracy, then loss, and named its
policies and their failures after its target of 99% accuracy.
"""

import json
from pathlib import Path
import unittest

from lab.domain.calibration import choose, rate_groups
from lab.domain.spec import describe
from lab.infrastructure.loader import load

from test_experiments import EXPERIMENTS

LEGACY = json.loads((Path(__file__).parent / "fixtures" / "legacy_rates.json").read_text())
STUDY = load(EXPERIMENTS / "mqar_sparsemax")
POLICIES = {"best": "best", "first99": "first", "stable99": "stable"}
STATUSES = {"selected": "selected", "99_percent_unreached": "target_unreached",
            "99_percent_not_sustained": "target_not_sustained"}


def label(run):
    return f"{run['attention']}-n{run['length']}-lr{run['learning_rate']:.8g}"


def candidate(run):
    """What the selection reads of an archived run, at the updates of its experiment's epochs."""
    experiment = STUDY.experiments[label(run)]
    assert experiment.optimizer.lr == run["learning_rate"]
    milestone, best = run["milestone"], run["best_validation"]
    return {"label": label(run), "lr": run["learning_rate"], "rank": (best["accuracy"], -best["loss"]),
            "crossing": None if milestone is None else {
                "step": milestone["epoch"] * experiment.evaluate.every,
                "training_seconds": milestone["training_seconds"],
                "sustained_to_end": milestone["sustained_to_end"]}}


class CalibrationTests(unittest.TestCase):
    def test_the_file_groups_its_runs_by_everything_but_the_rate(self):
        groups = rate_groups({label: describe(experiment) for label, experiment in STUDY.experiments.items()})
        rates = [label.rsplit("-", 1)[1] for label in groups[0]]
        self.assertEqual(groups, [[f"{attention}-n{length}-{rate}" for rate in rates]
                                  for attention in ("softmax", "sparsemax") for length in (64, 128, 256, 512)])

    def test_each_policy_selects_what_the_comparison_selected(self):
        for policy, groups in LEGACY["selections"].items():
            for group in groups:
                with self.subTest(policy=policy, attention=group["attention"], length=group["length"]):
                    runs = [run for run in LEGACY["runs"]
                            if (run["attention"], run["length"]) == (group["attention"], group["length"])]
                    found = choose([candidate(run) for run in runs], POLICIES[policy])
                    expected = {"status": STATUSES[group["status"]]}
                    if group["learning_rate"] is not None:
                        expected |= {"label": label(group), "lr": group["learning_rate"]}
                    self.assertEqual(found, expected)

    def test_the_policies_disagree_on_the_archived_runs(self):
        chosen = {policy: [group["learning_rate"] for group in groups]
                  for policy, groups in LEGACY["selections"].items()}
        self.assertEqual(len({tuple(rates) for rates in chosen.values()}), 3)
        with self.assertRaisesRegex(ValueError, "Unknown rate selection policy"):
            choose([], "last")


if __name__ == "__main__":
    unittest.main()
