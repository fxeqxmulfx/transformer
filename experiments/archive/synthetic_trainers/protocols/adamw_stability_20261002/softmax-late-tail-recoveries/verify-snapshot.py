"""Independently recompute the late prefix's episodes and complete-bin frequencies."""

import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def verify(directory, replay=None):
    directory = Path(directory)
    hashes = read(directory / "artifact-hashes.json")["files"]
    files = {str(p.relative_to(directory)) for p in directory.rglob("*")
             if p.is_file() and p.name != "artifact-hashes.json"}
    assert files == set(hashes)
    assert all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    plan = read(directory / "frozen-root-plan.json")
    native = read(directory / "captured-native-case-plan.json")
    recipe = next(r for r in plan["recipes"] if r["name"] == "adamw-softmax")
    assert native["config"] == recipe["config"] and native["config"]["steps"] == 300000
    assert native["source_hashes"] == plan["training_source_hashes"] and native["corpus"] == plan["corpus"]
    history = [json.loads(line) for line in (directory / "canonical-prefix.jsonl").read_text().splitlines()]
    assert [p["step"] for p in history] == list(range(0, 280001, 250))
    reference = [json.loads(line) for line in (directory / "reference-non-time-prefix.jsonl").read_text().splitlines()]
    assert [{k: v for k, v in p.items() if k not in ("training_seconds", "wall_seconds", "diagnostic_seconds")}
            for p in history] == reference
    criterion = plan["criterion"]; target = criterion["target"]
    passing = lambda p: min(p["train"]["accuracy"], p["heldout"]["accuracy"]) >= target
    assessment = read(directory / "assessment.json")
    assert assessment["long_confirmation"]["onset"] == 95750
    tail = [p for p in history if p["step"] >= 250000]
    assert len(tail) == assessment["tail_observations"] == 121
    assert [p["step"] for p in tail if not passing(p)] == assessment["tail_failures"] == [274000, 275500]
    assert not assessment["complete_canonical_history"] and not assessment["persistent_final_performance"]
    post = [p for p in history if p["step"] >= 95750]
    episodes = []; active = None
    for point in post:
        if not passing(point):
            if active is None:
                active = {"first_failure": point["step"], "failed_observations": 0, "minimum_accuracy": 1.0}
            active["last_failure"] = point["step"]; active["failed_observations"] += 1
            active["minimum_accuracy"] = min(active["minimum_accuracy"], point["train"]["accuracy"], point["heldout"]["accuracy"])
        elif active is not None:
            active["first_recovery"] = point["step"]
            active["sampled_steps_until_first_recovery"] = point["step"] - active["first_failure"]
            episodes.append(active); active = None
    assert active is None and len(episodes) == 6
    metrics = read(directory / "recovery-metrics.json")["metrics"]["joint"]
    assert metrics["episode_count"] == len(episodes)
    assert all(all(stored[key] == value for key, value in actual.items())
               for stored, actual in zip(metrics["episodes_after_long_onset"], episodes))
    assert [e["first_failure"] for e in episodes[-2:]] == [274000, 275500]
    assert [e["first_recovery"] for e in episodes[-2:]] == [274250, 275750]
    assert [e["sampled_steps_until_first_recovery"] for e in episodes[-2:]] == [250, 250]
    for window in metrics["fixed_tail_windows"][:3]:
        points = [p for p in history if window["start"] <= p["step"] < window["stop"]]
        failed = sum(not passing(p) for p in points)
        onsets = sum(window["start"] <= e["first_failure"] < window["stop"] for e in episodes)
        assert len(points) == window["observations"] == 40 and window["complete_window"]
        assert window["failed_observations"] == failed and window["failure_fraction"] == failed / 40
        assert window["new_episode_onsets"] == window["episode_onsets_per_10000_updates"] == onsets
    assert [w["new_episode_onsets"] for w in metrics["fixed_tail_windows"][:3]] == [0, 0, 2]
    triplets = read(directory / "boundary-triplets.json")
    canonical = {p["step"]: p for p in triplets["canonical"]}
    neighbors = {p["step"]: p for p in triplets["neighbors"]}
    assert canonical[274000]["last_batch_size"] == 512 and canonical[275500]["last_batch_size"] == 48
    assert neighbors[273999]["last_batch_size"] == 48
    assert not passing(neighbors[273999]) and not passing(neighbors[274001])
    assert passing(neighbors[275499]) and not passing(neighbors[275501])
    assert passing(neighbors[274249]) and passing(neighbors[275749]) and not passing(neighbors[275751])
    assert all(p["heldout"]["EOS_accuracy"] == 1 for p in [*canonical.values(), *neighbors.values()])
    replayed = 0
    if replay is not None:
        replay = Path(replay)
        names = ("assessment.json", "boundary-triplets.json", "canonical-prefix.jsonl", "capture-snapshot.py",
            "captured-native-case-plan.json", "frozen-root-plan.json", "recovery-metrics.json",
            "recovery-series.json", "reference-non-time-prefix.jsonl")
        assert all((directory / name).read_bytes() == (replay / name).read_bytes() for name in names)
        a = read(directory / "validation.json"); b = read(replay / "validation.json")
        assert {k: v for k, v in a.items() if k != "captured_utc"} == {k: v for k, v in b.items() if k != "captured_utc"}
        replayed = len(names)
    assert "torch" not in sys.modules
    return {"artifact_hashes_valid": True, "independent_episode_and_bin_recomputation_passed": True,
        "canonical_observations": 1121, "tail_observations": 121, "tail_failures": [274000, 275500],
        "complete_10k_tail_bin_failed_observations": [0, 0, 2], "total_sampled_recovered_episodes": 6,
        "replay_byte_exact_core_files": replayed, "recovery_time_and_continuity_between_evaluations_known": False,
        "Torch_imported": False, "scientific_run_complete": False}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
