"""Durable raw measurements and reproducible three-seed summaries."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_results(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(results, seeds):
    groups = defaultdict(list)
    for result in results:
        if result["phase"] == "final":
            groups[result["attention"], result["method"]].append(result)
    summaries = []
    for (attention, method), runs in groups.items():
        successful = [r for r in runs if r["status"] == "ok"]
        values = [r["test_loss"] for r in successful]
        complete = len(runs) == len(seeds) and {r["seed"] for r in successful} == set(seeds)
        average = lambda field: statistics.mean(r[field] for r in successful) if successful else None
        summaries.append({"attention": attention, "method": method, "lr": runs[0]["lr"],
                          "successful_seeds": len(successful), "expected_seeds": len(seeds),
                          "complete": complete, "test_loss_mean": statistics.mean(values) if values else None,
                          "test_loss_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                          "test_ppl": math.exp(statistics.mean(values)) if values else None,
                          "validation_loss_mean": average("validation_loss"),
                          "train_seconds_mean": average("train_seconds"),
                          "optimizer_ms_mean": average("optimizer_ms"),
                          "peak_memory_mib": max((r["peak_memory_mib"] for r in successful), default=None),
                          "guard_acceptance_mean": average("guard_acceptance")
                          if successful and successful[0]["guard_acceptance"] is not None else None})
    summaries.sort(key=lambda r: (r["attention"], not r["complete"], r["test_loss_mean"] or math.inf))
    return summaries


def winners(summaries):
    result = {}
    for row in summaries:
        if row["complete"] and row["attention"] not in result:
            result[row["attention"]] = row
    return result


def write_summary(directory, results, seeds):
    summaries = summarize(results, seeds)
    write_json(directory / "summary.json", {"winners": winners(summaries), "methods": summaries})
    if summaries:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries
