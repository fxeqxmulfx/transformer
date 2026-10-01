"""Summaries of best-validation checkpoints with realized budgets and stops."""

from collections import Counter, defaultdict
import csv
import math
import statistics

from experiments.optimizer_benchmark.storage import write_json


def summarize(rows, seeds):
    groups = defaultdict(list)
    for row in rows:
        groups[row["attention"], row["method"]].append(row)
    summaries = []
    for (attention, method), group in groups.items():
        usable = [r for r in group if r["status"] in {"ok", "recovered"} and r["test_loss"] is not None]
        values = [r["test_loss"] for r in usable]
        complete = len(group) == len(seeds) and {r["seed"] for r in usable} == set(seeds)
        average = lambda field: statistics.mean(r[field] for r in usable) if usable else None
        summaries.append({"attention": attention, "method": method, "lr": group[0]["lr"],
                          "complete": complete, "usable_seeds": len(usable), "expected_seeds": len(seeds),
                          "test_loss_mean": statistics.mean(values) if values else None,
                          "test_loss_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                          "test_ppl": math.exp(statistics.mean(values)) if values else None,
                          "validation_loss_mean": average("validation_loss"),
                          "actual_steps_mean": average("actual_steps"), "best_step_mean": average("best_step"),
                          "train_seconds_mean": average("train_seconds"), "total_seconds_mean": average("total_seconds"),
                          "optimizer_ms_mean": average("optimizer_ms"),
                          "peak_memory_mib": max((r["peak_memory_mib"] for r in usable), default=None),
                          "recovered_runs": sum(r["status"] == "recovered" for r in group),
                          "capped_runs": sum(r["stop_reason"] == "max_steps" for r in group),
                          "stop_reasons": dict(Counter(r["stop_reason"] for r in group))})
    return sorted(summaries, key=lambda r: (r["attention"], not r["complete"],
                                           r["test_loss_mean"] if r["test_loss_mean"] is not None else math.inf))


def write_summary(directory, rows, seeds):
    summaries = summarize(rows, seeds)
    winners = {}
    for row in summaries:
        if row["complete"] and row["attention"] not in winners:
            winners[row["attention"]] = row
    write_json(directory / "summary.json", {"winners": winners, "methods": summaries})
    if summaries:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries
