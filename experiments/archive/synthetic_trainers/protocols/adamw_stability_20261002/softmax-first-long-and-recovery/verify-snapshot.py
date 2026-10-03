"""Independently verify the saved partial witness using only the standard library."""

import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(directory, replay=None):
    directory = Path(directory)
    hashes = read(directory / "artifact-hashes.json")["files"]
    files = {str(p.relative_to(directory)) for p in directory.rglob("*")
             if p.is_file() and p.name != "artifact-hashes.json"}
    assert files == set(hashes)
    assert all(sha(directory / p) == value for p, value in hashes.items())
    plan = read(directory / "frozen-root-plan.json")
    native = read(directory / "captured-native-case-plan.json")
    recipe = next(r for r in plan["recipes"] if r["name"] == "adamw-softmax")
    assert native["config"] == recipe["config"] and native["config"]["steps"] == 300000
    assert native["source_hashes"] == plan["training_source_hashes"]
    assert native["corpus"] == plan["corpus"]
    history = [json.loads(line) for line in (directory / "canonical-prefix.jsonl").read_text().splitlines()]
    assert [p["step"] for p in history] == list(range(0, 105751, 250))
    reference = [json.loads(line) for line in (directory / "reference-non-time-prefix.jsonl").read_text().splitlines()]
    without_time = lambda p: {k: v for k, v in p.items()
                             if k not in ("training_seconds", "wall_seconds", "diagnostic_seconds")}
    assert [without_time(p) for p in history] == reference
    criterion = plan["criterion"]
    target = criterion["target"]
    passing = lambda p: p["train"]["accuracy"] >= target and p["heldout"]["accuracy"] >= target
    first_target = next(p["step"] for p in history if p["heldout"]["accuracy"] >= target)
    runs = []; streak = []
    for p in history:
        if p["step"] < first_target and p["train"]["accuracy"] >= target and p["heldout"]["accuracy"] <= criterion["heldout_ceiling"]:
            streak.append(p)
        elif streak:
            runs.append(streak); streak = []
    if streak:
        runs.append(streak)
    plateau = max(runs, key=len)
    assessment = read(directory / "assessment.json")
    assert assessment["plateau"] == {"start": plateau[0]["step"], "end": plateau[-1]["step"], "observations": len(plateau)}
    assert len(plateau) >= criterion["plateau_observations"]
    assert plateau[-1]["step"] - plateau[0]["step"] >= criterion["plateau_steps"]
    streak = []
    for p in history:
        streak = [*streak, p] if passing(p) else []
        if len(streak) == criterion["confirmation_observations"]:
            break
    long = assessment["long_confirmation"]
    assert long["onset"] == streak[0]["step"] == 95750
    assert long["confirmed"] == streak[-1]["step"] == 100500
    for name in ("training_seconds", "wall_seconds"):
        assert long[name] == streak[-1][name]
    post = [p for p in history if p["step"] >= long["onset"]]
    failures = [p for p in post if not passing(p)]
    assert [p["step"] for p in failures] == [105250, 105500]
    recovery = next(p for p in post if p["step"] > failures[-1]["step"] and passing(p))
    metrics = read(directory / "recovery-metrics.json")["metrics"]["joint"]
    assert metrics["post_long_onset_observations"] == len(post)
    assert metrics["post_long_onset_failure_fraction"] == len(failures) / len(post)
    assert metrics["episode_count"] == 1
    episode = metrics["episodes_after_long_onset"][0]
    assert episode["first_recovery"] == recovery["step"] == 105750
    assert episode["sampled_steps_until_first_recovery"] == recovery["step"] - failures[0]["step"] == 500
    assert episode["minimum_accuracy"] == min(min(p["train"]["accuracy"], p["heldout"]["accuracy"]) for p in failures)
    assert not assessment["complete_canonical_history"] and assessment["tail_observations"] == 0
    assert all(w["observations"] == 0 and w["failure_fraction"] is None for w in metrics["fixed_tail_windows"])
    triplets = read(directory / "boundary-triplets.json")
    boundaries = (100500, 105250, 105500, 105750)
    assert triplets["canonical"] == [p for p in history if p["step"] in boundaries]
    selected = {s + offset for s in boundaries for offset in (-1, 0, 1)}
    assert {p["step"] for p in triplets["neighbors"]} == selected - set(boundaries)
    neighbors = {p["step"]: p for p in triplets["neighbors"]}
    assert not passing(neighbors[105501]) and passing(neighbors[105749])
    assert [p["step"] for p in triplets["gradients"]] == sorted(selected)
    assert [p["step"] for p in triplets["tensor_diagnostics"]] == sorted(selected)
    assert all(p["learning_rate"] == 0.0003 and p["batch_size"] == 512 for p in triplets["gradients"])
    replayed = 0
    if replay is not None:
        replay = Path(replay)
        names = ("assessment.json", "boundary-triplets.json", "canonical-prefix.jsonl", "capture-snapshot.py",
                 "captured-native-case-plan.json", "frozen-root-plan.json", "recovery-metrics.json", "reference-non-time-prefix.jsonl")
        assert all((directory / name).read_bytes() == (replay / name).read_bytes() for name in names)
        a = read(directory / "validation.json"); b = read(replay / "validation.json")
        assert {k: v for k, v in a.items() if k != "captured_utc"} == {k: v for k, v in b.items() if k != "captured_utc"}
        replayed = len(names)
    assert "torch" not in sys.modules
    return {"offline_artifact_hashes_valid": True, "independent_phase_and_episode_recomputation_passed": True,
            "canonical_observations": len(history), "replay_byte_exact_core_files": replayed,
            "A_A_observed_non_time_metrics_exact": True, "first_sampled_recovery": recovery["step"],
            "sampled_recovery_updates": 500, "actual_first_between_evaluation_recovery_time_known": False,
            "final_window_observations": 0, "scientific_run_complete": False, "Torch_imported": False}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
