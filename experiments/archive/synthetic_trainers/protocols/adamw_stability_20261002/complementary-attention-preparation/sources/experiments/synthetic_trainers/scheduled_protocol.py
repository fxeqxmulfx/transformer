"""Freeze and execute two full native AdamW budgets with one declared schedule change."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from .attention_report import verify_pair as verify_reference
from .paper_reproduction.diagnostics import DiagnosticsConfig
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import ROOT
from .persistence import PersistenceConfig
from .runtime import write_json
from .scheduled_layout import current_sources, digest, validate_pair_plan
from .scheduled_training import ScheduledRunConfig, train, training_sources
from .stability import validate_complete, validate_run
from .stability_comparison import hashes
from .stability_protocol import environment, paper_fingerprints
from .stability_recovery import describe
from .stability_recovery_series import timeline


def verify_live(directory, expected):
    directory = Path(directory)
    if json.loads((directory/"plan.json").read_text()) != expected:
        raise ValueError("Frozen schedule plan changed")
    validate_pair_plan(expected)
    if current_sources() != expected["source_hashes"] or training_sources() != expected["training_source_hashes"]:
        raise ValueError("Frozen schedule sources changed")
    base = expected["recipes"][0]["config"]
    if environment(base["device"]) != expected["environment"] or paper_fingerprints() != expected["papers"]:
        raise ValueError("Frozen environment or local manuscripts changed")
    if make_corpus(base["prime"],base["train_fraction"],base["data_seed"]).summary() != expected["corpus"]:
        raise ValueError("Frozen schedule corpus changed")
    if any(digest(ROOT/name) != value for name,value in expected["Lean_specification_hashes"].items()):
        raise ValueError("Frozen reference Lean sources changed")
    audit=ROOT/"experiments/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-preparation/validation.json"
    if digest(audit) != expected["Lean_audit_validation_sha256"]:
        raise ValueError("Frozen reference Lean audit changed")
    if expected["scientific_run"]:
        reference=directory/"reference"
        if hashes(reference) != expected["reference_hashes"] or verify_reference(reference) != expected["reference_complete_outcome"]:
            raise ValueError("The complete normalizer reference changed")
    return expected


def freeze_pair(directory, base, diagnostics, criterion, *, reference=None):
    directory=Path(directory)
    if directory.exists():
        raise FileExistsError("Schedule calibration requires a fresh directory")
    if type(base) is not ScheduledRunConfig or base.learning_rate_schedule != "constant":
        raise ValueError("The explicitly tagged constant control is required")
    scientific=base.device.startswith("cuda")
    if scientific and reference is None:
        raise ValueError("Scientific scheduling waits for the complete normalizer reference")
    reference_outcome=verify_reference(reference) if reference is not None else None
    if scientific and (not reference_outcome["complete_pair"] or not reference_outcome["scientific_run"]
            or reference_outcome["outcome"]["control"]["stable_grokking"]):
        raise ValueError("Scheduling requires a complete failed primary reference; a passing softmax enters independent confirmation")
    reference_plan=json.loads((Path(reference)/"plan.json").read_text()) if reference is not None else None
    protocol=ROOT/"experiments/synthetic_trainers/protocols/adamw_stability_20261002"
    audit=json.loads((protocol/"sparsemax-preparation/validation.json").read_text())
    names=("adamw-constant","adamw-cosine-tail")
    plan={"stage":"conditional_fixed_schedule_calibration_pair","scientific_run":scientific,
        "scope":"conditional_native_AdamW_schedule_adaptation; no_architecture_claim" if scientific else "CPU_schedule_pair_fixture; no_learning_result",
        "recipes":[{"name":name,"config":asdict(replace(base,learning_rate_schedule=kind)),
                    "changed_mechanism":"unchanged_softmax_constant_control" if index==0 else "learning_rate_schedule_only"}
                   for index,(name,kind) in enumerate(zip(names,("constant","cosine_tail")))],
        "execution_order":list(names),"planned_runs":2,"maximum_updates_per_run":300000,
        "campaign_architecture_claim_allowed":False,"criterion":asdict(criterion),
        "instrumentation":asdict(diagnostics),"recovery_window_steps":10000,
        "final_window":[base.steps-criterion.tail_steps,base.steps],
        "source_hashes":current_sources(),"training_source_hashes":training_sources(),
        "environment":environment(base.device),"papers":paper_fingerprints(),
        "corpus":make_corpus(base.prime,base.train_fraction,base.data_seed).summary(),
        "Lean_specification_hashes":{k:v for k,v in audit["source_hashes"].items() if k.startswith("src/")},
        "Lean_audit_validation_sha256":digest(protocol/"sparsemax-preparation/validation.json"),
        "reference_plan":reference_plan,"reference_complete_outcome":reference_outcome,
        "reference_hashes":hashes(reference) if reference is not None else None,
        "frozen_utc":datetime.now(timezone.utc).isoformat(),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "protocol":"two_fresh_full_budgets; fixed_schedule_only; no_target_stopping; no_dropped_failures",
        "selection":"only_complete_scientific_stable_grokking_can_enter_a_new_six_case_confirmation",
        "memory_scope":"per_case_maximum_of_observed_snapshots_and_execution_segments; clear_unused_CUDA_cache_before_each_case"}
    validate_pair_plan(plan)
    directory.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=directory.name+"-",dir=directory.parent) as tmp:
        output=Path(tmp)/"stage";output.mkdir()
        if reference is not None:
            shutil.copytree(reference,output/"reference")
        write_json(output/"plan.json",plan)
        verify_live(output,plan)
        output.rename(directory)
    return plan


def run_pair(directory, plan, *, progress=None):
    directory=Path(directory)
    verify_live(directory,plan)
    completed=[]
    with (directory/".lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this schedule pair") from error
        state={"status":"running","planned_runs":2,"completed_runs":0,"current_run":None,"runs":[]}
        try:
            for recipe in plan["recipes"]:
                verify_live(directory,plan)
                name=recipe["name"]
                state={**state,"completed_runs":len(completed),"current_run":name,"runs":list(completed)}
                write_json(directory/"state.json",state)
                run=directory/name
                peaks_path=run/"observed-peaks.json"
                peaks=json.loads(peaks_path.read_text()) if peaks_path.exists() else {
                    "peak_cuda_allocated_bytes":None,"peak_cuda_reserved_bytes":None}

                def observe(row):
                    if not row.get("diagnostic_probe") and recipe["config"]["device"].startswith("cuda"):
                        for key,measure in (("peak_cuda_allocated_bytes",torch.cuda.max_memory_allocated),
                                            ("peak_cuda_reserved_bytes",torch.cuda.max_memory_reserved)):
                            peaks[key]=max(peaks[key] or 0,measure(recipe["config"]["device"]))
                        write_json(peaks_path,peaks)
                    if progress:
                        progress({"run":name,**row})

                path=run/"measurements.json"
                if path.exists():
                    report=json.loads(path.read_text())
                else:
                    resuming=(run/"plan.json").exists()
                    if resuming:
                        validate_run(json.loads((run/"plan.json").read_text()),recipe,plan)
                    if recipe["config"]["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
                    report=train(ScheduledRunConfig(**recipe["config"]),run,resume=resuming,
                        diagnostics=DiagnosticsConfig(**plan["instrumentation"]),progress=observe)
                    for key,value in peaks.items():
                        if value is not None:
                            report[key]=max(report[key],value)
                    write_json(path,report)
                assessment=validate_complete(report,recipe,plan)
                criterion=PersistenceConfig(**plan["criterion"])
                write_json(run/"recovery-metrics.json",describe(report,criterion,window_steps=10000))
                write_json(run/"recovery-series.json",timeline(report,criterion,window_steps=10000))
                completed.append({"name":name,"result":str(path.relative_to(directory)),"assessment":assessment,
                    "training_seconds":report["training_seconds"],"wall_seconds":report["wall_seconds"],"final":report["final"]})
                if progress:
                    progress({"event":"completed_run",**completed[-1]})
        except BaseException as error:
            write_json(directory/"state.json",{**state,"status":"failed","completed_runs":len(completed),
                "runs":completed,"error":repr(error)})
            raise
        final={**state,"status":"complete","completed_runs":2,"current_run":None,"runs":completed}
        write_json(directory/"state.json",final)
    return final
