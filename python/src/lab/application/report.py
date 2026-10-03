"""What a study's runs recorded, read back: each run's state and result, and what its records say.

Nothing here trains or changes a run. The analyses read a run's stored
description, so a run whose file has since raised its budget is read
against the budget it trained to.
"""

from ..domain.benchmarks import ModularDivision
from ..domain.collapse import collapse, largest_gradients
from ..domain.experiment import require_continuation
from ..domain.phases import phases
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


def report_study(study, labels, runs: Runs):
    """Each chosen run's state, latest observation and result, with the analyses its benchmark reads.

    A run directory holding another experiment than the file defines is an
    error, as it is for training.
    """
    reports = {}
    for label, experiment in study.select(labels):
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
        reports[label] = found
    return reports
