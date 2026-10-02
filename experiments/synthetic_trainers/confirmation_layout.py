"""Portable confirmation layout and calibration gate; no training dependencies."""

import hashlib
import json
from pathlib import Path

from .stability_comparison import hashes, verify_comparison


# The previous modular campaign used model seeds 1/2/3 and data seed 1;
# exploratory calibration uses model/data seed 0. New confirmation excludes both.
RESERVED_MODEL_SEEDS = (0, 1, 2, 3)
RESERVED_DATA_SEEDS = (0, 1)


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


def case_manifest(plan, recipe):
    keys = ("criterion", "instrumentation", "training_source_hashes", "environment",
            "papers", "frozen_utc", "git_commit")
    return {**{key: plan[key] for key in keys}, "recipes": [recipe], "planned_runs": 1,
            "corpus": recipe["corpus"], "source_hashes": plan["driver_source_hashes"],
            "stage": "independent_confirmation_case", "confirmation_plan_sha256": plan_hash(plan),
            "protocol": "full_fixed_budget; no_target_stopping; retain_failures",
            "scope": "one_case_of_a_prospectively_frozen_crossed_confirmation"}


def selected_recipe(calibration, name):
    calibration = Path(calibration)
    summary = verify_comparison(calibration)
    if name not in summary["stable_grokking_calibration_recipes"]:
        raise ValueError("Confirmation requires a passing recipe from a complete calibration comparison")
    manifest = json.loads((calibration / "plan.json").read_text())
    recipe = next(row for row in manifest["recipes"] if row["name"] == name)
    if recipe["config"]["model"] != "gptmini" or recipe["config"]["optimizer"] != "amsgradw":
        raise ValueError("Confirmation must retain raw AMSGradW and GPTMini")
    return manifest, recipe


def validate_plan(plan, calibration):
    if (plan["stage"] != "independent_confirmation"
            or plan["success_rule"] != "all_frozen_cases_pass_stable_grokking; no_dropped_failures"):
        raise ValueError("Invalid confirmation stage or success rule")
    manifest, selected = selected_recipe(calibration, plan["selected_recipe"])
    if plan["calibration_hashes"] != hashes(Path(calibration)):
        raise ValueError("Frozen calibration evidence changed")
    for key in ("criterion", "instrumentation", "environment", "papers", "training_source_hashes"):
        if plan[key] != manifest[key]:
            raise ValueError(f"Confirmation {key} differs from calibration")
    if plan["driver_source_hashes"] != manifest["source_hashes"]:
        raise ValueError("Confirmation training/driver sources differ from calibration")
    model_seeds, data_seeds = plan["model_seeds"], plan["data_seeds"]
    for seeds, minimum, key, reserved in ((model_seeds, 3, "seed", RESERVED_MODEL_SEEDS),
                                          (data_seeds, 2, "data_seed", RESERVED_DATA_SEEDS)):
        used = {row["config"][key] for row in manifest["recipes"]} | set(reserved)
        if (len(seeds) < minimum or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 or seed in used for seed in seeds)):
            raise ValueError("Confirmation requires distinct seeds excluded from calibration")
    expected = [(seed, data) for data in data_seeds for seed in model_seeds]
    if plan["planned_runs"] != len(expected) or len(plan["recipes"]) != len(expected):
        raise ValueError("Confirmation matrix is incomplete")
    corpus_by_data = {}
    for row, (seed, data) in zip(plan["recipes"], expected):
        config = {**selected["config"], "seed": seed, "data_seed": data}
        if row["name"] != f"seed-{seed}-data-{data}" or row["config"] != config:
            raise ValueError("Confirmation changed more than the independent seeds")
        corpus = row["corpus"]
        for key in manifest["corpus"]:
            if key not in ("data_seed", "train_fingerprint", "heldout_fingerprint") and corpus[key] != manifest["corpus"][key]:
                raise ValueError("Confirmation corpus dimensions changed")
        if corpus["data_seed"] != data or corpus_by_data.setdefault(data, corpus) != corpus:
            raise ValueError("Model seeds do not share their declared data split")
    train_hashes = {corpus["train_fingerprint"] for corpus in corpus_by_data.values()}
    heldout_hashes = {corpus["heldout_fingerprint"] for corpus in corpus_by_data.values()}
    if (len(train_hashes) != len(data_seeds) or len(heldout_hashes) != len(data_seeds)
            or manifest["corpus"]["train_fingerprint"] in train_hashes
            or manifest["corpus"]["heldout_fingerprint"] in heldout_hashes):
        raise ValueError("Confirmation reused a calibration or repeated data fingerprint")
    return selected
