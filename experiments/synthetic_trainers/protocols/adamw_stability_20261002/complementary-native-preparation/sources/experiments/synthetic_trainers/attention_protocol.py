"""Freeze and serially execute the user-directed softmax/sparsemax exploration.

Convexifying Transformers, Section 4 provides the task context; the sparsemax
replacement follows Transformer.GPTMini.Convex. Both full budgets and every
failure remain visible; the pair does not replace independent confirmations.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from .attention_layout import current_sources, digest, validate_pair_plan
from .attention_training import AttentionRunConfig, train, training_sources
from .paper_reproduction.diagnostics import DiagnosticsConfig
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import ROOT
from .persistence import PersistenceConfig
from .runtime import write_json
from .stability import validate_complete, validate_run
from .stability_comparison import hashes, verify_comparison
from .stability_protocol import environment, paper_fingerprints
from .stability_recovery import describe
from .stability_recovery_series import timeline


def verify_live(directory, expected):
    directory = Path(directory)
    if json.loads((directory / "plan.json").read_text()) != expected:
        raise ValueError("Frozen normalizer plan changed")
    validate_pair_plan(expected)
    if expected["source_hashes"] != current_sources() or expected["training_source_hashes"] != training_sources():
        raise ValueError("Frozen normalizer sources changed")
    if expected["environment"] != environment(expected["recipes"][0]["config"]["device"]):
        raise ValueError("Frozen normalizer environment changed")
    if expected["papers"] != paper_fingerprints():
        raise ValueError("Frozen normalizer manuscripts changed")
    base = expected["recipes"][0]["config"]
    if make_corpus(base["prime"],base["train_fraction"],base["data_seed"]).summary() != expected["corpus"]:
        raise ValueError("Frozen normalizer corpus fingerprints changed")
    if any(digest(ROOT/name) != value for name, value in expected["Lean_specification_hashes"].items()):
        raise ValueError("The audited Lean specification changed")
    audit = ROOT / "experiments/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-preparation/validation.json"
    if digest(audit) != expected["Lean_audit_validation_sha256"]:
        raise ValueError("The retained Lean audit changed")
    if expected["scientific_run"]:
        reference = directory / "reference"
        if hashes(reference) != expected["reference_hashes"]:
            raise ValueError("Complete reference evidence changed")
        if verify_comparison(reference) != expected["reference_complete_summary"]:
            raise ValueError("Verified reference summary changed")
    return expected


def freeze_pair(directory, base, diagnostics, criterion, *, reference=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Normalizer pair requires a fresh output directory")
    if type(base) is not AttentionRunConfig or base.attention_normalization != "softmax":
        raise ValueError("The unchanged tagged softmax configuration is required")
    scientific = base.device.startswith("cuda")
    if scientific and reference is None:
        raise ValueError("Scientific sparsemax waits for the verified completed reference")
    reference_summary = verify_comparison(reference) if reference is not None else None
    reference_plan = json.loads((Path(reference)/"plan.json").read_text()) if reference is not None else None
    protocol = ROOT / "experiments/synthetic_trainers/protocols/adamw_stability_20261002"
    audited = json.loads((protocol / "sparsemax-preparation/validation.json").read_text())
    lean_hashes = {k:v for k,v in audited["source_hashes"].items() if k.startswith("src/")}
    plan = {"stage": "exploratory_attention_normalizer_pair", "scientific_run": scientific,
        "scope": "user_directed_exploratory_pair; no_repeatable_architecture_claim" if scientific else "CPU_pipeline_fixture; no_learning_result",
        "recipes": [{"name": "adamw-"+name, "config": asdict(replace(base, attention_normalization=name)),
                     "changed_mechanism": "unchanged_model_control" if name == "softmax" else "attention_normalization_only"}
                    for name in ("softmax", "sparsemax")],
        "planned_runs": 2, "maximum_updates_per_run": 300000, "campaign_architecture_claim_allowed": False,
        "execution_order": ["adamw-sparsemax","adamw-softmax"],
        "improvement_rule": {"quality_metric": "final_exhaustive_complete_RHS_accuracy",
                             "minimum_accuracy_gain": .01, "minimum_time_ratio": 1.05,
                             "timing_requires_both_training_and_wall": True},
        "criterion": asdict(criterion), "instrumentation": asdict(diagnostics),
        "recovery_window_steps": 10000, "final_window": [base.steps-criterion.tail_steps, base.steps],
        "source_hashes": current_sources(), "training_source_hashes": training_sources(),
        "environment": environment(base.device), "papers": paper_fingerprints(),
        "corpus": make_corpus(base.prime, base.train_fraction, base.data_seed).summary(),
        "Lean_specification_hashes": lean_hashes,
        "Lean_audit_validation_sha256": digest(protocol/"sparsemax-preparation/validation.json"),
        "reference_plan": reference_plan, "reference_complete_summary": reference_summary,
        "reference_hashes": hashes(reference) if reference is not None else None,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "two_fresh_full_budgets; normalizer_only; no_target_stopping; no_dropped_failures",
        "memory_scope": "per_recipe_maximum_of_available_canonical_snapshots_and_completed_execution_segments; clear_unused_CUDA_cache_before_each_recipe",
        "independent_confirmation_gate": "all_six_fresh_benchmark_cases_required; this_pair_does_not_open_that_gate"}
    validate_pair_plan(plan)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=directory.name+"-", dir=directory.parent) as temporary:
        output = Path(temporary)/"stage"
        output.mkdir()
        if reference is not None:
            shutil.copytree(reference, output/"reference")
        write_json(output/"plan.json", plan)
        verify_live(output, plan)
        output.rename(directory)
    return plan


def run_pair(directory, plan, *, progress=None):
    directory = Path(directory)
    verify_live(directory, plan)
    completed = []
    with (directory/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this normalizer pair") from error
        recipes={recipe["name"]:recipe for recipe in plan["recipes"]}
        for name in plan["execution_order"]:
            recipe=recipes[name]
            verify_live(directory, plan)
            state = {"status": "running", "planned_runs": 2, "completed_runs": len(completed),
                     "current_run": name, "runs": completed}
            write_json(directory/"state.json", state)
            run = directory/name
            peaks_path = run/"observed-peaks.json"
            peaks = json.loads(peaks_path.read_text()) if peaks_path.exists() else {
                "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}

            def observe(row):
                if not row.get("diagnostic_probe") and recipe["config"]["device"].startswith("cuda"):
                    for key, measure in (("peak_cuda_allocated_bytes",torch.cuda.max_memory_allocated),
                                         ("peak_cuda_reserved_bytes",torch.cuda.max_memory_reserved)):
                        peaks[key] = max(peaks[key] or 0,measure(recipe["config"]["device"]))
                    write_json(peaks_path,peaks)
                if progress:
                    progress({"run":name,**row})

            try:
                path = run/"measurements.json"
                if path.exists():
                    report = json.loads(path.read_text())
                else:
                    resuming = (run/"plan.json").exists()
                    if resuming:
                        validate_run(json.loads((run/"plan.json").read_text()),recipe,plan)
                    if recipe["config"]["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
                    report = train(AttentionRunConfig(**recipe["config"]),run,resume=resuming,
                                   diagnostics=DiagnosticsConfig(**plan["instrumentation"]),progress=observe)
                    for key, value in peaks.items():
                        if value is not None:
                            report[key] = max(report[key],value)
                    write_json(path,report)
                assessment = validate_complete(report,recipe,plan)
                criterion = PersistenceConfig(**plan["criterion"])
                write_json(run/"recovery-metrics.json",describe(report,criterion,window_steps=10000))
                write_json(run/"recovery-series.json",timeline(report,criterion,window_steps=10000))
            except BaseException as error:
                write_json(directory/"state.json",{**state,"status":"failed","error":repr(error)})
                raise
            completed.append({"name":name,"result":str(path.relative_to(directory)),"assessment":assessment,
                              "training_seconds":report["training_seconds"],"wall_seconds":report["wall_seconds"],"final":report["final"]})
            if progress:
                progress({"event":"completed_run",**completed[-1]})
        final = {**state,"status":"complete","completed_runs":2,"current_run":None,"runs":completed}
        write_json(directory/"state.json",final)
    return final
