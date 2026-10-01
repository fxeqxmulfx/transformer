"""Summaries of best-validation checkpoints with realized budgets and stops."""

from collections import Counter, defaultdict
import csv
import math
import statistics

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import write_json


from gpt_mini.domain.ranking import summarize


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
