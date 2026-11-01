"""Freeze Basis, train complete states, and reserve all final test until closure."""

import argparse
from datetime import datetime, timezone
import json
import platform
import time
import traceback

import torch

from lab.domain.basis import BASIS
from lab.domain.experiment import require_continuation
from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.basis_residual import BasisTraining
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code, provenance
from lab.infrastructure.store import RunDirectory
from protocol import HERE, frozen, overlap, sha, sources, write


def freeze():
    if (HERE / "protocol.json").exists():
        raise FileExistsError("Basis comparison is already frozen")
    study, cases, data = load(HERE), [], {}
    # Each task/size has two paired seeds; reverse the order in the second seed.
    order = tuple(f"{variant}-{size}-{task}-s{seed}"
                  for task in ("depth", "recall", "parity") for size in ("small", "large")
                  for seed in (0, 1)
                  for variant in (("canonical", "adaptive") if seed == 0 else ("adaptive", "canonical")))
    torch.set_num_threads(1)
    for label in order:
        spec = study.experiments[label]
        _, size, name, _ = label.split("-")
        mode = "easy" if size == "small" else "hard"
        if (mode, name) not in data:
            task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
            original = build_task(BASIS[mode][name][0], spec.seeds.data, torch.device("cpu"))
            if task.fingerprints != original.fingerprints:
                raise ValueError("Original Basis splits changed")
            data[mode, name] = {"fingerprints": task.fingerprints, "overlap": overlap(task), "vocab": task.vocab}
        info = data[mode, name]
        model = build_model(spec.model, info["vocab"], spec.seeds.model)
        cases.append({"label": label, "experiment": describe(spec), "data": info,
                      "parameters": sum(parameter.numel() for parameter in model.parameters())})
    write(HERE / "protocol.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "engine_sources": code(), "cohort_sources": sources(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "source_gpt_commit": "df1e9f7", "selection": "standard small/easy and large/hard Basis, model seeds0/1, original data seed1; no hyperparameter search",
        "stop": "first observation with sequence_accuracy>=.99; validation/length-128 for hard depth, validation otherwise; free generation including EOS for parity",
        "continuation": "100000 updates is an initial complete-state chunk, not proof of failure; promising cases continue before final test",
        "test_policy": "open test only after all cases cross99 or an explicit numerical failure or censored continuation decision is recorded; no test-based architecture selection",
        "interpretation": "report every paired crossing and every unresolved case; improvement on modular division alone does not establish improvement on Basis",
        "production_rule": "retain the current production architecture if a candidate loses a Basis success at its paired control's crossing or has unresolved task regressions; summarize size/task effects rather than pooling away failures",
        "deviations_from_historical_basis": ["current normalized source GPT rather than GPTMini", "Schedule-Free AdamW rather than AdamW", "CUDA fullgraph max-autotune and math SDPA rather than compiled CPU", "evaluation batch64 rather than256", "first crossing endpoint rather than best-state restore", "100000-update initial chunks rather than original short budgets"],
    })
    print("Frozen24 Basis runs with unchanged full original split fingerprints", flush=True)


def progress(label, validation, row):
    print(json.dumps({"label": label, "step": row["step"], "validation": row[validation]["sequence_accuracy"],
                      "loss": row[validation]["loss"], "training_seconds": row["training_seconds"]}), flush=True)


def run():
    declaration = frozen()
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("Frozen runtime changed")
    phase_path = HERE / "runs/phase.json"
    phase = json.loads(phase_path.read_text()) if phase_path.exists() else {
        "status": "learning", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": {}}
    if phase["status"] == "learning_closed":
        raise RuntimeError("Learning is closed; do not extend after final test")
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
        try:
            directory.begin(describe(spec), provenance(spec.execution.device), study.source)
            trainer = BasisTraining(spec, directory,
                lambda row: progress(label, spec.benchmark.validation_name, row), ResidualCompiledStepper)
            if trainer.task.fingerprints != case["data"]["fingerprints"]:
                raise ValueError("Frozen Basis data changed")
        except Exception:
            record.update(status="preparation_failure", traceback=traceback.format_exc(),
                          elapsed_seconds=time.monotonic() - start)
            write(phase_path, phase)
            raise
        record.update(status="learning", preparation_seconds=time.monotonic() - start)
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
    complete = len(phase["cases"]) == len(declaration["cases"]) and all(
        record["status"] in ("validation99", "numerical_failure") for record in phase["cases"].values())
    phase["status"] = "learning_closed" if complete else "awaiting_continuation"
    if complete:
        phase["closed_utc"] = datetime.now(timezone.utc).isoformat()
    write(phase_path, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "report"))
    action = parser.parse_args().action
    if action == "report":
        from report import report
        report()
    else:
        {"freeze": freeze, "run": run}[action]()
