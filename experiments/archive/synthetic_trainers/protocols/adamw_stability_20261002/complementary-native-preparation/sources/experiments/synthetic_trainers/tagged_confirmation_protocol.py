"""Freeze and execute all six fresh tagged softmax repeats with unchanged native trainers."""

from dataclasses import asdict,replace
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from .attention_training import AttentionRunConfig,train as normalizer_train,training_sources as normalizer_sources
from .paper_reproduction.diagnostics import DiagnosticsConfig
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import ROOT
from .runtime import write_json
from .scheduled_training import ScheduledRunConfig,train as schedule_train,training_sources as schedule_sources
from .stability import validate_run,validate_complete
from .stability_comparison import hashes
from .stability_protocol import environment,paper_fingerprints
from .tagged_confirmation_layout import current_sources,digest,select,case_manifest,validate_plan
from .tagged_confirmation_report import save_run,verify_archive


def trainer(mode):
    return (AttentionRunConfig,normalizer_train,normalizer_sources) if mode=="normalizer" else (ScheduledRunConfig,schedule_train,schedule_sources)


def verify_live(directory,plan):
    directory=Path(directory)
    if json.loads((directory/"plan.json").read_text())!=plan or current_sources()!=plan["source_hashes"]:
        raise ValueError("Frozen tagged confirmation plan or sources changed")
    validate_plan(plan,directory/"calibration")
    if (trainer(plan["training_mode"])[2]() != plan["training_source_hashes"]
            or environment(plan["recipes"][0]["config"]["device"])!=plan["environment"]
            or paper_fingerprints()!=plan["papers"]):
        raise ValueError("The tagged native trainer, environment or local manuscripts changed")
    if any(digest(ROOT/name)!=value for name,value in plan["Lean_specification_hashes"].items()):
        raise ValueError("The frozen Lean specification changed")
    audit=ROOT/"experiments/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-preparation/validation.json"
    if digest(audit)!=plan["Lean_audit_validation_sha256"]:
        raise ValueError("The frozen Lean audit changed")
    for recipe in plan["recipes"]:
        config=recipe["config"]
        if make_corpus(config["prime"],config["train_fraction"],config["data_seed"]).summary()!=recipe["corpus"]:
            raise ValueError("A frozen independent corpus changed")
        if json.loads((directory/"cases"/recipe["name"]/"plan.json").read_text())!=case_manifest(plan,recipe):
            raise ValueError("Frozen tagged confirmation case plan changed")
    return plan


def freeze(directory,calibration,name,*,CPU_fixture=False,render=True):
    directory,calibration=Path(directory),Path(calibration)
    if directory.exists():
        raise FileExistsError("Tagged confirmation requires a fresh output")
    origin,selected,mode,outcome=select(calibration,name,CPU_fixture=CPU_fixture)
    factory,_,sources=trainer(mode);base=factory(**selected["config"])
    if sources()!=origin["training_source_hashes"] or any(current_sources().get(k)!=v for k,v in origin["source_hashes"].items()):
        raise ValueError("Confirmation must retain every calibration source and the exact tagged native trainer")
    if environment(base.device)!=origin["environment"] or paper_fingerprints()!=origin["papers"]:
        raise ValueError("Confirmation must retain the calibration environment and manuscripts")
    recipes=[]
    for data in (2,3):
        corpus=make_corpus(base.prime,base.train_fraction,data).summary()
        for seed in (4,5,6):
            recipes.append({"name":f"seed-{seed}-data-{data}","config":asdict(replace(base,seed=seed,data_seed=data)),
                "corpus":corpus,"changed_mechanism":"independent_model_and_data_seeds_only"})
    plan={"stage":"independent_confirmation","selected_recipe":name,"training_mode":mode,
        "scientific_run":origin["scientific_run"],"CPU_fixture":CPU_fixture,"maximum_updates_per_run":300000,
        "model_seeds":[4,5,6],"data_seeds":[2,3],"planned_runs":6,"recipes":recipes,"archive_plots":render,
        "output_directory":str(directory),
        "source_hashes":current_sources(),"driver_source_hashes":origin["source_hashes"],
        "training_source_hashes":origin["training_source_hashes"],"calibration_hashes":hashes(calibration),
        "calibration_complete_outcome":outcome,
        **{key:origin[key] for key in ("environment","papers","criterion","instrumentation",
            "Lean_specification_hashes","Lean_audit_validation_sha256")},
        "frozen_utc":datetime.now(timezone.utc).isoformat(),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "success_rule":"all_frozen_cases_pass_stable_grokking; no_dropped_failures",
        "protocol":"six_fresh_full_serial_budgets; retain_config_tags_native_state_and_every_failure",
        "memory_scope":"per_case_maximum_of_observed_snapshots_and_execution_segments; clear_unused_CUDA_cache_before_each_case",
        "statistical_scope":"crossed_initializations_and_data_splits; not_independent_IID_runs_or_a_confidence_interval"}
    validate_plan(plan,calibration)
    directory.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=directory.name+"-",dir=directory.parent) as temporary:
        output=Path(temporary)/"stage";output.mkdir();shutil.copytree(calibration,output/"calibration")
        write_json(output/"plan.json",plan)
        for recipe in recipes:
            case=output/"cases"/recipe["name"];case.mkdir(parents=True)
            write_json(case/"plan.json",case_manifest(plan,recipe))
        verify_live(output,plan);output.rename(directory)
    return plan


def run_confirmation(directory,plan,*,progress=None):
    directory=Path(directory);verify_live(directory,plan);completed=[]
    with (directory/".lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this tagged confirmation") from error
        state={"status":"running","planned_runs":6,"completed_runs":0,"current_run":None,"runs":[]}
        try:
            for recipe in plan["recipes"]:
                verify_live(directory,plan);name=recipe["name"]
                state={**state,"completed_runs":len(completed),"current_run":name,"runs":list(completed)}
                write_json(directory/"state.json",state)
                case=directory/"cases"/name;manifest=case_manifest(plan,recipe)
                run=case/name;result=run/"measurements.json";archive=directory/"archives"/name
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
                if not result.exists():
                    resuming=(run/"plan.json").exists()
                    if resuming:
                        validate_run(json.loads((run/"plan.json").read_text()),recipe,manifest)
                    if recipe["config"]["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
                    factory,execute,_=trainer(plan["training_mode"])
                    report=execute(factory(**recipe["config"]),run,resume=resuming,
                        diagnostics=DiagnosticsConfig(**plan["instrumentation"]),
                        progress=observe)
                    for key,value in peaks.items():
                        if value is not None:
                            report[key]=max(report[key],value)
                    write_json(result,report)
                validate_complete(json.loads(result.read_text()),recipe,manifest)
                if archive.exists():
                    summary=verify_archive(archive)
                    if json.loads((archive/"plan.json").read_text())!=manifest or any((archive/filename).read_bytes()!=(run/filename).read_bytes()
                            for filename in ("measurements.json","diagnostics.jsonl","probes.jsonl","gradients.jsonl")):
                        raise ValueError("Existing tagged case archive differs from its complete raw records")
                else:
                    summary=save_run(case,name,archive,render=plan["archive_plots"])
                completed.append({"name":name,"assessment":summary["assessment"],"archive":f"archives/{name}"})
                if progress:
                    progress({"event":"completed_confirmation_case",**completed[-1]})
        except BaseException as error:
            write_json(directory/"state.json",{**state,"status":"failed","completed_runs":len(completed),"runs":completed,"error":repr(error)})
            raise
        state={**state,"status":"complete","completed_runs":6,"current_run":None,"runs":completed,
            "repeatable_stable_benchmark":bool(plan["scientific_run"] and all(row["assessment"]["stable_grokking"] for row in completed))}
        write_json(directory/"state.json",state)
    return state
