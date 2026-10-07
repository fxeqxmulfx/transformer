"""Freeze a compact, explicitly partial snapshot of the fresh observations.

From python/: uv run --locked python ../experiments/grokking_progress/summarize.py.
Raw source hashes accompany causal event times. Later accuracy is used only
for evaluating an earlier signal, never for constructing that signal.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from lab.domain.grokking import norm_progress, progress
from lab.infrastructure.loader import load

STUDY = Path(__file__).resolve().parent


def records(path):
    data = path.read_bytes()
    complete = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.splitlines()], hashlib.sha256(complete).hexdigest()


def main():
    runs = {}
    for label, experiment in load(STUDY).select([]):
        root = STUDY / "runs" / label
        if not (root / "history.jsonl").exists():
            runs[label] = {"status": "not_started", "budget": experiment.budget.updates}
            continue
        history, history_hash = records(root / "history.jsonl")
        diagnostics, diagnostic_hash = records(root / "diagnostics.jsonl")
        evidence = progress(history)
        events = evidence["first_events"]
        first99 = next((row["step"] for row in history if row["heldout"]["answer_accuracy"] >= .99), None)
        formation = events["structure_forming"]
        item = {"status": "finished" if (root / "result.json").exists() else "unfinished",
                "through_update": history[-1]["step"], "budget": experiment.budget.updates,
                "history_sha256": history_hash, "diagnostics_sha256": diagnostic_hash,
                "first_events": events, "first_heldout_99": first99,
                "lead_to_first_heldout_99": first99 - formation if first99 is not None and formation is not None else None,
                "at_first_structure_signal": next((row for row in history if row["step"] == formation), None),
                "latest": history[-1], "latest_norms": norm_progress(diagnostics)["latest"],
                "criterion": evidence["criterion"], "training_device": experiment.execution.device,
                "threads": experiment.execution.threads}
        probes = root / "internal_checkpoint_probes.jsonl"
        if probes.exists():
            snapshots, probe_hash = records(probes)
            item["internal_checkpoint_probes_sha256"] = probe_hash
            item["internal_checkpoints"] = [{key: row[key] for key in
                                             ("step", "checkpoint_sha256", "observation_device", "training_device",
                                              "heldout_invariant_energy_fraction", "raw_heldout_loss",
                                              "heldout_restricted_loss", "internal", "norms")}
                                            for row in snapshots]
        runs[label] = item
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "runs": runs,
              "scope": "fresh_observations_to_named_updates; incomplete_budgets_are_not_final_results",
              "prediction_status": "one_known_seed_positive_case; complete_negative_and_independent_controls_pending"}
    (STUDY / "fresh_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label, row in runs.items():
        print(label, row["status"], row.get("through_update"), row.get("first_events"))


if __name__ == "__main__":
    main()
