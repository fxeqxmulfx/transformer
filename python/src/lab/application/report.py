"""What a study's runs recorded, read back: each run's state and result, and what its records say.

Nothing here trains or changes a run. The analyses read a run's stored
description, so a run whose file has since raised its budget is read
against the budget it trained to. Runs that differ in their learning rate
alone are compared once each has trained the budget its file sets.
"""

from ..domain.benchmarks import ModularDivision
from ..domain.calibration import POLICIES, choose, rate_groups
from ..domain.collapse import collapse, largest_gradients
from ..domain.experiment import require_continuation
from ..domain.phases import phases
from ..domain.spec import describe
from ..domain.stability import persistence, recovery
from .ports import Run, Runs


def status(result):
    """Unfinished without a result; finished when the result reached the budget; stopped otherwise."""
    if result is None:
        return "unfinished"
    return "finished" if result["stop"]["reason"] == "budget" else "stopped"


def stability(run: Run, history, budget, every, finished):
    """The phases, persistence, recovery and failure neighborhoods of a modular division run.

    The analyses of the delayed-generalization stability protocols, under
    their fixed criterion. Recovery reads only a run whose budget spans the
    criterion's tail and whose cadence divides its windows; otherwise it
    records why not.
    """
    diagnostics = run.records("diagnostics")
    neighborhoods = collapse(history, run.records("probes"), diagnostics)
    try:
        recovered = recovery(history, budget, every, finished)
    except ValueError as error:
        recovered = {"not_applicable": str(error)}
    return {"phases": phases(history), "persistence": persistence(history, budget, every, finished),
            "recovery": recovered, "collapse": neighborhoods,
            "largest_gradients": largest_gradients(diagnostics, neighborhoods["confirmed_step"])}


def candidate(label, experiment, history, result):
    """What a finished run offers the rate selection: its rate, the rank of its best observation, and, when its
    benchmark marks milestones, its first observation at 99%."""
    benchmark = experiment.benchmark
    found = {"label": label, "lr": experiment.optimizer.lr,
             "rank": benchmark.rank(result["best"][benchmark.selection])}
    if "milestones" in result:
        reached = result["milestones"]["99"]
        seconds = {row["step"]: row["training_seconds"] for row in history}
        found["crossing"] = None if reached is None else {
            "step": reached["step"], "training_seconds": seconds[reached["step"]],
            "sustained_to_end": reached["sustained_to_end"]}
    return found


def select_rates(chosen, candidates):
    """For each group of chosen runs that differ in their rate alone, the run each policy selects.

    Only a benchmark with a selection split is calibrated, and the crossing
    policies need its milestones too. A group waits until each of its runs
    has trained its file's budget.
    """
    groups = rate_groups({label: describe(experiment) for label, experiment in chosen
                          if experiment.benchmark.selection is not None})
    selections = []
    for labels in groups:
        incomplete = [label for label in labels if label not in candidates]
        if incomplete:
            selections.append({"labels": labels, "incomplete": incomplete})
            continue
        group = [candidates[label] for label in labels]
        policies = POLICIES if all("crossing" in member for member in group) else ("best",)
        selections.append({"labels": labels, "selected": {policy: choose(group, policy) for policy in policies}})
    return selections


def report_study(study, labels, runs: Runs):
    """Each chosen run's state, latest observation and result, with the analyses its benchmark reads, and the
    rate selection among them.

    A run directory holding another experiment than the file defines is an
    error, as it is for training.
    """
    reports, candidates = {}, {}
    chosen = study.select(labels)
    for label, experiment in chosen:
        run = runs.open(study.name, label)
        stored = run.description()
        if stored is None:
            reports[label] = {"status": "not_started"}
            continue
        require_continuation(stored, experiment)
        history, result = run.records("history"), run.result()
        budget = stored["budget"]["updates"]
        found = {"status": status(result), "budget": budget, "latest": history[-1] if history else None,
                 "result": result}
        if history and isinstance(experiment.benchmark, ModularDivision):
            found |= stability(run, history, budget, stored["evaluate"]["every"], found["status"] == "finished")
        if (found["status"] == "finished" and budget == experiment.budget.updates
                and experiment.benchmark.selection is not None):
            candidates[label] = candidate(label, experiment, history, result)
        reports[label] = found
    return {"runs": reports, "rate_selection": select_rates(chosen, candidates)}
