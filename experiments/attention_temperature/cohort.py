"""Freeze and run the isolated learned-temperature intervention in current GPT."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import platform
import shutil
import time
import traceback

import torch

from lab.domain.experiment import require_continuation
from lab.domain.adaptive_residual import CanonicalResidual
from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.basis_residual import BasisTraining
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code, provenance
from lab.infrastructure.store import RunDirectory
from protocol import HERE, frozen, sha, sources, write

TERMINAL = ("validation99", "numerical_failure", "censored")


def freeze():
    if (HERE / "protocol.json").exists():
        raise FileExistsError("The temperature study is already frozen")
    study = load(HERE)
    previous = json.loads((HERE.parent / "basis_residual/protocol.json").read_text())
    results = json.loads((HERE.parent / "basis_residual/results.json").read_text())
    cases = []
    torch.set_num_threads(1)
    for label, spec in study.experiments.items():
        seed = spec.seeds.model
        reference_label = f"canonical-large-recall-s{seed}"
        reference = next(c for c in previous["cases"] if c["label"] == reference_label)
        measured = next(c for c in results["cases"] if c["label"] == reference_label)
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        if task.fingerprints != reference["data"]["fingerprints"]:
            raise ValueError("The original hard-recall data changed")
        canonical = build_model(CanonicalResidual(**vars(spec.model)), task.vocab, seed)
        model = build_model(spec.model, task.vocab, seed)
        before, after = dict(canonical.named_parameters()), dict(model.named_parameters())
        for name, value in before.items():
            torch.testing.assert_close(value, after[name], rtol=0, atol=0)
        extras = set(after) - set(before)
        if extras != {f"blocks.{i}.sa.log_gain" for i in range(spec.model.depth)}:
            raise ValueError("Unexpected parameter difference")
        if any(torch.count_nonzero(after[name]).item() for name in extras):
            raise ValueError("Every initial score gain must be one")
        cases.append({"label": label, "experiment": describe(spec), "fingerprints": task.fingerprints,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "added_parameters": sum(after[name].numel() for name in extras),
                      "common_initial_parameters_exact": True,
                      "current_reference": {"label": reference_label, "status": measured["status"],
                                            "step": measured["step"], "checkpoint_sha256": measured["checkpoint_sha256"]}})
    names = ("basis_residual/protocol.json", "basis_residual/results.json", "basis_control/protocol.json",
             "basis_control/analysis/gptmini-hard-recall-s0-step12200.json",
             "basis_control/analysis/gptmini-hard-recall-s1-step22350.json")
    write(HERE / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "engine_sources": code(), "cohort_sources": sources(),
        "reference_files": {name: sha(HERE.parent / name) for name in names},
        "python": platform.python_version(), "torch": str(torch.__version__),
        "current_source": "diff_gpt df1e9f7; matched Basis reference23154349",
        "motivation": "GPTMini seed0 solves under matched Schedule-Free; both successful and slow seeds learn larger temperatures, so association is not causation",
        "intervention": "Only24 zero-initialized per-head log-gains, multiplying queries after Q/K RMS and RoPE",
        "stop": "first99% exact validation sequences",
        "continuation": "100000 updates is an initial boundary; preserve full state and explicitly extend promising cases",
        "test_policy": "evaluate only after both cases are terminal; original public Basis rows are reused, not a fresh benchmark across the research program",
        "promotion": "This is a screen. No production replacement until broader Basis/arithmetic confirmation, direct GPT regression tests and same-assistant review."})
    print("Frozen both temperature candidates with exact common initialization and original data", flush=True)


def run():
    declaration = frozen()
    if (HERE / "results.json").exists():
        raise ValueError("Final test is already open; learning cannot resume")
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("The frozen runtime changed")
    path = HERE / "runs/phase.json"
    phase = json.loads(path.read_text()) if path.exists() else {
        "status": "learning", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": {}}
    if phase["status"] in ("learning_closed", "awaiting_continuation"):
        raise ValueError("Learning is closed or needs an explicit continuation decision")
    study = load(HERE)
    for case in declaration["cases"]:
        label = case["label"]
        record = phase["cases"].setdefault(label, {})
        if record.get("status") in TERMINAL or record.get("status") == "budget":
            continue
        spec = study.experiments[label]
        if describe(spec) != case["experiment"]:
            raise ValueError("The frozen recipe changed")
        directory = RunDirectory(HERE / "runs" / label)
        if directory.description() is not None:
            require_continuation(directory.description(), spec)
            # Preserve interrupted logs before the engine rewinds unsaved rows.
            archive = directory.path / "interruptions" / str(time.time_ns())
            archive.mkdir(parents=True)
            for source in [*directory.path.glob("*.json*"), directory.path / "checkpoint.pt"]:
                if source.exists():
                    shutil.copyfile(source, archive / source.name)
            record.setdefault("interruptions", []).append({"files": {
                p.name: sha(p) for p in sorted(archive.iterdir())}, "path": str(archive.relative_to(HERE))})
        record["status"], phase["status"] = "preparing", "learning"
        write(path, phase)
        start = time.monotonic()
        try:
            directory.begin(describe(spec), provenance(spec.execution.device), study.source)
            trainer = BasisTraining(spec, directory, lambda row: print(json.dumps({
                "label": label, "step": row["step"],
                "sequence": row["validation"]["sequence_accuracy"],
                "token": row["validation"]["token_accuracy"]}), flush=True), ResidualCompiledStepper)
            if trainer.task.fingerprints != case["fingerprints"]:
                raise ValueError("The frozen data changed")
            record.update(status="learning", preparation_seconds=time.monotonic() - start)
            write(path, phase)
            try:
                result = trainer()
                record.update(status=result["stop"]["reason"], stop=result["stop"])
            except FloatingPointError:
                record.update(status="numerical_failure", traceback=traceback.format_exc())
            record.update(elapsed_seconds=time.monotonic() - start,
                          checkpoint_sha256=sha(directory.path / "checkpoint.pt"))
            del trainer
            torch.cuda.empty_cache()
        except BaseException:
            record["status"], phase["status"] = "interrupted", "interrupted"
            record["traceback"] = traceback.format_exc()
            write(path, phase)
            raise
        write(path, phase)
    complete = all(r["status"] in TERMINAL for r in phase["cases"].values())
    phase["status"] = "learning_closed" if complete else "awaiting_continuation"
    if complete:
        phase["closed_utc"] = datetime.now(timezone.utc).isoformat()
    write(path, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/cohort.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "report":
            from report import report
            report()
        else:
            {"freeze": freeze, "run": run}[args.action]()
