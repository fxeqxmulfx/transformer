"""Capture the first frozen sparsemax tail violation from complete raw observations.

This is a partial scientific witness, not a completed run or an instability
episode. The unchanged full 300000-update budget and paired control continue.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.stability_integrity import expected_batches, validate_observation

PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
RAW = Path("experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k")
NAME = "adamw-sparsemax"
BOUNDARY = 250000

def rows(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The active writer may have one unfinished final line.
    return result

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def capture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("A scientific boundary witness requires a fresh destination")
    plan_blob = (RAW / "plan.json").read_bytes()
    manifest = json.loads(plan_blob)
    if manifest != json.loads((PROTOCOL / "attention-pair-plan.json").read_text()):
        raise ValueError("The live root differs from the committed scientific pair")
    recipe = next(row for row in manifest["recipes"] if row["name"] == NAME)
    config = recipe["config"]
    criterion = PersistenceConfig(**manifest["criterion"])
    if (not manifest["scientific_run"] or config["steps"] != 300000
            or config["attention_normalization"] != "sparsemax"
            or config["steps"] - criterion.tail_steps != BOUNDARY):
        raise ValueError("This witness requires the actual frozen scientific tail boundary")
    case = RAW / NAME
    raw_plan = json.loads((case / "plan.json").read_text())
    for key, expected in {"config": config, "source_hashes": manifest["training_source_hashes"],
            "corpus": manifest["corpus"], "instrumentation": manifest["instrumentation"],
            "torch": manifest["environment"]["torch"], "gpu": manifest["environment"]["gpu"]}.items():
        if raw_plan[key] != expected:
            raise ValueError("The active native case differs from the frozen scientific pair")
    for filename, expected in manifest["source_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("An active frozen Python source changed")
    for filename, expected in manifest["Lean_specification_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("An active frozen Lean source changed")
    audit = PROTOCOL / "sparsemax-preparation/validation.json"
    if digest(audit) != manifest["Lean_audit_validation_sha256"]:
        raise ValueError("The original Lean audit changed")
    history = [row for row in rows(case / "history.jsonl") if row["step"] <= BOUNDARY]
    if [row["step"] for row in history] != list(range(0, BOUNDARY + 1, config["eval_every"])):
        raise ValueError("The scientific boundary witness lacks its entire canonical prefix")
    point = history[-1]
    neighbors = [row for row in rows(case / "probes.jsonl") if row["step"] in (BOUNDARY - 1, BOUNDARY + 1)]
    if [row["step"] for row in neighbors] != [BOUNDARY - 1, BOUNDARY + 1]:
        raise ValueError("Both actual immediate neighbors are required")
    selected_steps = {BOUNDARY - 1, BOUNDARY, BOUNDARY + 1}
    gradients = []
    with (case / "gradients.jsonl").open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["step"] in selected_steps:
                gradients.append(row)
            if row["step"] > BOUNDARY + 1:
                break
    diagnostics = [row for row in rows(case / "diagnostics.jsonl") if row["step"] in selected_steps]
    if [row["step"] for row in gradients] != sorted(selected_steps) or [row["step"] for row in diagnostics] != sorted(selected_steps):
        raise ValueError("The boundary triplet lacks actual rates, gradients or tensor diagnostics")
    for observation in [*history, *neighbors]:
        validate_observation(observation, manifest, config)
        if observation["step"]:
            batch = expected_batches(config, observation["step"])
            if observation["last_batch_size"] != batch["batch_size"]:
                raise ValueError("The boundary batch is not the declared short-final policy")
    for row in gradients:
        if row["learning_rate"] != config["learning_rate"] or row["gradient_l2"] < 0:
            raise ValueError("Actual boundary gradients or rates are invalid")
    assessment = assess({"plan": raw_plan, "history": history}, criterion)
    if (assessment["complete_canonical_history"] or assessment["long_confirmation"] is not None
            or assessment["tail_failures"] != [BOUNDARY] or point["heldout"]["accuracy"] >= criterion.target):
        raise ValueError("The actual first-tail violation differs from this recorded partial conclusion")
    output.mkdir(parents=True)
    (output / "frozen-root-plan.json").write_bytes(plan_blob)
    write(output / "captured-native-case-plan.json", raw_plan)
    (output / "canonical-prefix.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    write(output / "boundary-witness.json", {"canonical": point, "neighbors": neighbors,
                                           "gradients": gradients, "tensor_diagnostics": diagnostics})
    write(output / "assessment.json", assessment)
    write(output / "validation.json", {"captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "partial_scientific_first_tail_violation; complete_budget_still_running",
        "scientific_run": True, "complete_run": False, "case": NAME,
        "boundary_step": BOUNDARY, "declared_total_updates": 300000,
        "entire_canonical_prefix_observations": len(history), "required_final_window": [250000, 300000],
        "criterion_target": criterion.target, "first_tail_heldout_accuracy": point["heldout"]["accuracy"],
        "first_tail_train_accuracy": point["train"]["accuracy"], "first_tail_EOS_accuracy": point["heldout"]["EOS_accuracy"],
        "cannot_meet_frozen_all_observations_final_tail": True,
        "first_long_confirmation": None, "post_confirmation_instability_episode_count": None,
        "all_43_Python_and_9_Lean_and_audit_unchanged": True,
        "paired_softmax_control_not_completed": True, "full_budget_continues_unchanged": True,
        "Torch_imported": "torch" in sys.modules})
    write(output / "artifact-hashes.json", {"files": {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact-hashes.json"}})
    return point

if __name__ == "__main__":
    point = capture(sys.argv[1])
    print(json.dumps({"step": point["step"], "train": point["train"]["accuracy"],
                      "heldout": point["heldout"]["accuracy"], "complete_run": False, "Torch_imported": "torch" in sys.modules}))
