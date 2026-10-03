"""Freeze and execute all six held-out native AdamW normalizer pairs serially."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from .architecture_cohort_layout import case_manifest, current_sources, digest, select, validate_plan, RULE, INITIALIZATION, STATISTICAL_SCOPE
from .architecture_cohort_report import save_case as save_run, verify_case as verify_archive
from .architecture_training import ArchitectureRunConfig, train, training_sources
from .paper_reproduction.diagnostics import DiagnosticsConfig
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import ROOT
from .runtime import write_json
from .stability import validate_complete, validate_run
from .stability_comparison import hashes
from .stability_protocol import environment, paper_fingerprints


def verify_live(directory, plan):
    directory = Path(directory)
    if json.loads((directory / "plan.json").read_text()) != plan or current_sources() != plan["source_hashes"]:
        raise ValueError("The frozen architecture plan or sources changed")
    validate_plan(plan, directory / "benchmark")
    if (training_sources() != plan["training_source_hashes"] or paper_fingerprints() != plan["papers"]
            or environment(plan["base_config"]["device"]) != plan["environment"]):
        raise ValueError("The native trainer, manuscripts or selected device environment changed")
    if any(digest(ROOT / name) != value for name, value in plan["Lean_specification_hashes"].items()):
        raise ValueError("The frozen Lean specification changed")
    audit = ROOT / "experiments/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-preparation/validation.json"
    if digest(audit) != plan["Lean_audit_validation_sha256"]:
        raise ValueError("The existing Lean audit receipt changed")
    for recipe in plan["recipes"]:
        config = recipe["config"]
        if make_corpus(config["prime"], config["train_fraction"], config["data_seed"]).summary() != recipe["corpus"]:
            raise ValueError("A frozen architecture corpus changed")
        if json.loads((directory / "cases" / recipe["name"] / "plan.json").read_text()) != case_manifest(plan, recipe):
            raise ValueError("A frozen paired case plan changed")
    return plan


def freeze(directory, benchmark, *, CPU_fixture=False, render=True):
    directory, benchmark = Path(directory), Path(benchmark)
    if directory.exists():
        raise FileExistsError("Architecture cohorts require a fresh output")
    origin, selected, outcome = select(benchmark, CPU_fixture=CPU_fixture)
    base = ArchitectureRunConfig(**selected)
    sources = current_sources()
    if any(sources.get(k) != v for k, v in origin["source_hashes"].items()):
        raise ValueError("The selected benchmark source version changed")
    if environment(base.device) != origin["environment"] or paper_fingerprints() != origin["papers"]:
        raise ValueError("Architecture pairs retain the benchmark environment and manuscripts")
    recipes, pairs = [], []
    for data in (4, 5):
        corpus = make_corpus(base.prime, base.train_fraction, data).summary()
        for seed in (7, 8, 9):
            name = f"seed-{seed}-data-{data}"
            pair = {"name": name, "model_seed": seed, "data_seed": data,
                    "control": name + "-softmax", "candidate": name + "-sparsemax"}
            roles = ("control", "candidate") if len(pairs) % 2 == 0 else ("candidate", "control")
            pairs.append(pair)
            for role in roles:
                norm = "softmax" if role == "control" else "sparsemax"
                recipes.append({"name": pair[role], "config": asdict(replace(base,
                    seed=seed, data_seed=data, attention_normalization=norm)), "corpus": corpus})
    plan = {"stage": "paired_architecture_cohort", "scientific_run": origin["scientific_run"],
        "CPU_fixture": CPU_fixture, "planned_pairs": 6, "planned_runs": 12, "base_config": asdict(base),
        "model_seeds": [7, 8, 9], "data_seeds": [4, 5], "recipes": recipes, "pairs": pairs,
        "execution_order": [r["name"] for r in recipes], "maximum_updates_per_run": 300000,
        "archive_plots": render, "changed_fields": ["attention_normalization"], "improvement_rule": RULE,
        "source_hashes": sources, "training_source_hashes": training_sources(),
        "benchmark_hashes": hashes(benchmark), "benchmark_complete_outcome": outcome,
        **{k: origin[k] for k in ("criterion", "instrumentation", "environment", "papers",
            "Lean_specification_hashes", "Lean_audit_validation_sha256")},
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(directory),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initialization_pairing": INITIALIZATION,
        "statistical_scope": STATISTICAL_SCOPE,
        "memory_scope": "per_case_maximum_of_canonical_snapshots_and_execution_segments; clear_unused_cache"}
    validate_plan(plan, benchmark)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=directory.name + "-", dir=directory.parent) as temporary:
        output = Path(temporary) / "stage"; output.mkdir()
        shutil.copytree(benchmark, output / "benchmark")
        write_json(output / "plan.json", plan)
        for recipe in recipes:
            case = output / "cases" / recipe["name"]; case.mkdir(parents=True)
            write_json(case / "plan.json", case_manifest(plan, recipe))
        verify_live(output, plan); output.rename(directory)
    return plan


def run_cohort(directory, plan, *, progress=None):
    directory = Path(directory); verify_live(directory, plan); completed = []
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this architecture cohort") from error
        state = {"status": "running", "planned_runs": 12, "completed_runs": 0, "current_run": None, "runs": []}
        try:
            for recipe in plan["recipes"]:
                verify_live(directory, plan); name = recipe["name"]
                state = {**state, "completed_runs": len(completed), "current_run": name, "runs": list(completed)}
                write_json(directory / "state.json", state)
                case = directory / "cases" / name; manifest = case_manifest(plan, recipe)
                run = case / name; result = run / "measurements.json"; archive = directory / "archives" / name
                peaks_path = run / "observed-peaks.json"
                peaks = json.loads(peaks_path.read_text()) if peaks_path.exists() else {
                    "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
                def observe(row):
                    if not row.get("diagnostic_probe") and recipe["config"]["device"].startswith("cuda"):
                        for key, measure in (("peak_cuda_allocated_bytes", torch.cuda.max_memory_allocated),
                                             ("peak_cuda_reserved_bytes", torch.cuda.max_memory_reserved)):
                            peaks[key] = max(peaks[key] or 0, measure(recipe["config"]["device"]))
                        write_json(peaks_path, peaks)
                    if progress:
                        progress({"run": name, **row})
                if not result.exists():
                    resuming = (run / "plan.json").exists()
                    if resuming:
                        validate_run(json.loads((run / "plan.json").read_text()), recipe, manifest)
                    if recipe["config"]["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
                    report = train(ArchitectureRunConfig(**recipe["config"]), run, resume=resuming,
                                   diagnostics=DiagnosticsConfig(**plan["instrumentation"]), progress=observe)
                    for key, value in peaks.items():
                        if value is not None:
                            report[key] = max(report[key], value)
                    write_json(result, report)
                validate_complete(json.loads(result.read_text()), recipe, manifest)
                if archive.exists():
                    summary = verify_archive(archive)
                    if (json.loads((archive / "plan.json").read_text()) != manifest or any(
                            (archive / filename).read_bytes() != (run / filename).read_bytes()
                            for filename in ("measurements.json", "diagnostics.jsonl", "probes.jsonl", "gradients.jsonl"))):
                        raise ValueError("Existing architecture archive differs from complete raw records")
                else:
                    summary = save_run(case, name, archive, render=plan["archive_plots"])
                completed.append({"name": name, "assessment": summary["assessment"], "archive": f"archives/{name}"})
                if progress:
                    progress({"event": "completed_architecture_case", **completed[-1]})
            verify_live(directory, plan)
        except BaseException as error:
            write_json(directory / "state.json", {**state, "status": "failed", "completed_runs": len(completed),
                "runs": completed, "error": repr(error)})
            raise
        state = {**state, "status": "complete", "completed_runs": 12, "current_run": None, "runs": completed}
        write_json(directory / "state.json", state)
    return state
