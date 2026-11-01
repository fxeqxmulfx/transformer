"""Freeze the matched cohort, run/resume complete states, then open final test."""

import argparse
from datetime import datetime, timezone
import json
import platform
import time
import traceback

import torch

from lab.domain.experiment import require_continuation
from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code, provenance
from lab.infrastructure.store import RunDirectory
from protocol import HERE, CrossingTraining, frozen, same_data, sha, sources, write


def freeze():
    if (HERE / "protocol.json").exists():
        raise FileExistsError("The matched protocol is already frozen")
    study = load(HERE)
    cases = []
    # Reverse order in the second pair to expose simple thermal/order confounds.
    order = tuple(f"{variant}-p{prime}-s{seed}" for prime in (97, 193)
                  for seed in (11, 17)
                  for variant in (("canonical", "adaptive") if seed == 11 else ("adaptive", "canonical")))
    for label in order:
        spec = study.experiments[label]
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        cases.append({"label": label, "experiment": describe(spec), "task": task.metadata,
                      "parameters": sum(p.numel() for p in model.parameters())})
    write(HERE / "protocol.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "engine_sources": code(), "cohort_sources": sources(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "source_gpt_commit": "df1e9f7", "selection": "P97/P193 with two matched model seeds each, unchanged data/batches11; no search",
        "stop": "first canonical validation>=99%, every1000 updates; current endpoint, not best checkpoint",
        "continuation": "100000 is a checkpointed chunk ceiling, not success or an automatic rejection; promising uncrossed cases need continuation",
        "test_policy": "sealed until every case has crossed99 or has an explicitly recorded numerical failure",
        "decision": "promote only if both seeds on both primes cross99 earlier than their current-GPT controls, and every source/gradient/cache regression passes; final test is reported, not used for selection",
        "synthetic_scope": "input-dependent branch amplitude; no claim that all grokking obstructions are removed",
    })
    print("Frozen eight fresh matched transformer runs", flush=True)


def run():
    declaration = frozen()
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("Frozen runtime changed")
    phase_path = HERE / "runs/phase.json"
    phase = json.loads(phase_path.read_text()) if phase_path.exists() else {
        "status": "learning", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": {}}
    if phase["status"] == "learning_closed":
        raise RuntimeError("Learning is closed; no extension after final test")
    study = load(HERE)
    for case in declaration["cases"]:
        label = case["label"]
        previous = phase["cases"].get(label, {})
        if previous.get("status") in ("validation99", "numerical_failure"):
            continue
        spec = study.experiments[label]
        if describe(spec) != case["experiment"]:
            raise ValueError("Frozen recipe changed")
        directory = RunDirectory(HERE / "runs" / label)
        if directory.description() is not None:
            require_continuation(directory.description(), spec)
        record = {**previous, "status": "preparing"}
        phase["cases"][label] = record
        write(phase_path, phase)
        start = time.monotonic()
        directory.begin(describe(spec), provenance(spec.execution.device), study.source)
        try:
            trainer = CrossingTraining(spec, directory,
                lambda row: print(json.dumps({"label": label, **row}, allow_nan=False), flush=True),
                ResidualCompiledStepper)
            if not same_data(trainer.task.metadata, case["task"]):
                raise ValueError("Frozen split changed")
        except Exception:
            record.update(status="preparation_failure", traceback=traceback.format_exc(),
                          elapsed_seconds=time.monotonic() - start)
            write(phase_path, phase)
            raise
        record["preparation_seconds"] = time.monotonic() - start
        record["status"] = "learning"
        write(phase_path, phase)
        try:
            result = trainer()
            record.update(status=result["stop"]["reason"], stop=result["stop"])
        except FloatingPointError:
            record.update(status="numerical_failure", traceback=traceback.format_exc())
        record.update(elapsed_seconds=time.monotonic() - start,
                      checkpoint_sha256=sha(directory.path / "checkpoint.pt"))
        write(phase_path, phase)
        print(json.dumps({"label": label, **record}), flush=True)
        del trainer
        torch.cuda.empty_cache()
    if all(record["status"] in ("validation99", "numerical_failure") for record in phase["cases"].values()):
        phase.update(status="learning_closed", closed_utc=datetime.now(timezone.utc).isoformat())
    else:
        phase["status"] = "awaiting_continuation"
    write(phase_path, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "report"))
    arguments = parser.parse_args()
    if arguments.action == "report":
        from report import report
        report()
    else:
        {"freeze": freeze, "run": run}[arguments.action]()
