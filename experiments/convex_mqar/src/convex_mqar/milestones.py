"""First observed validation crossings from measured cumulative epoch timings.

These observations have epoch resolution. Training time includes lazy kernel
compilation and excludes validation, checkpoints, pauses, data generation,
and work discarded by a checkpoint resume. Training plus validation is a sum
of the recorded timers, not a reconstruction of total elapsed wall time.
"""

import math


THRESHOLDS = (50, 75, 90, 95, 99)


def _seconds(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def summarize_epochs(records, thresholds=THRESHOLDS):
    """Use the first measured crossing; keep unreached thresholds as null.

Integer query counts determine crossing when available. A later accuracy drop
does not change the first observation, and is exposed by sustained_to_end.
Reject duplicate/gapped epochs and decreasing cumulative time rather than
silently mixing retried or incomplete histories.
"""
    if not records:
        raise ValueError("An epoch history is required")
    if len(set(thresholds)) != len(thresholds) or any(not 0 < p <= 100 for p in thresholds):
        raise ValueError("Accuracy thresholds must be distinct percentages in (0, 100]")
    validation_seconds = 0.0
    observations = []
    previous_training = 0.0
    for expected, row in enumerate(records, start=1):
        if row["epoch"] != expected:
            raise ValueError("Epochs must be contiguous, unique, and start at 1")
        training = _seconds(row["training_seconds"], "training_seconds")
        if training < previous_training:
            raise ValueError("Cumulative training time decreased")
        previous_training = training
        validation = row["validation"]
        accuracy = validation["accuracy"]
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Validation accuracy must be finite and between 0 and 1")
        if "correct" in validation and "query_count" in validation:
            correct, count = validation["correct"], validation["query_count"]
            if count <= 0 or not 0 <= correct <= count or abs(accuracy - correct / count) > 1e-12:
                raise ValueError("Validation accuracy disagrees with query counts")
        if "seconds" not in validation or validation_seconds is None:
            validation_seconds = None
        else:
            validation_seconds += _seconds(validation["seconds"], "validation_seconds")
        observations.append({
            "epoch": row["epoch"], "accuracy": accuracy,
            "training_seconds": training,
            "training_plus_validation_seconds": None if validation_seconds is None else
                training + validation_seconds,
            "compiled_loss": row.get("compiled_loss", False), "validation": validation})

    def reached(observation, percent):
        validation = observation["validation"]
        if "correct" in validation and "query_count" in validation:
            return 100 * validation["correct"] >= percent * validation["query_count"]
        return observation["accuracy"] >= percent / 100

    milestones = {}
    for percent in thresholds:
        index = next((i for i, row in enumerate(observations) if reached(row, percent)), None)
        if index is None:
            milestones[str(percent)] = None
            continue
        row = observations[index]
        previous = observations[index - 1] if index else None
        milestones[str(percent)] = {
            "epoch": row["epoch"], "validation_accuracy": row["accuracy"],
            "training_seconds": row["training_seconds"],
            "training_plus_validation_seconds": row["training_plus_validation_seconds"],
            "previous_validation_epoch": None if previous is None else previous["epoch"],
            "previous_validation_accuracy": None if previous is None else previous["accuracy"],
            "previous_training_seconds": None if previous is None else previous["training_seconds"],
            "execution_modes_until_crossing": sorted({
                "compiled" if r["compiled_loss"] else "eager" for r in observations[:index + 1]}),
            "sustained_to_end": all(reached(r, percent) for r in observations[index:])}
    peak = max(records, key=lambda row: (row["validation"]["accuracy"], -row["validation"]["loss"]))
    return {"epochs_observed": len(records), "best_epoch": peak["epoch"],
            "best_validation": peak["validation"],
            "last_validation_accuracy": observations[-1]["accuracy"],
            "total_training_seconds": observations[-1]["training_seconds"],
            "total_training_plus_validation_seconds": observations[-1]["training_plus_validation_seconds"],
            "milestones": milestones}
