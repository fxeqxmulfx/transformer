"""Preflight, validation-only rate screening and the complete patience comparison."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.data import TextData
from experiments.optimizer_benchmark.runner import log_progress
from experiments.optimizer_benchmark.storage import read_results, write_json
from experiments.patience_benchmark.stopping import StopConfig
from .protocol import (ROOT,METHODS,RATES,baseline,check_initial,digest,
                       preflight_artifacts,prior_artifacts,source_hashes)
from .report import write_report
from .training import train_one
from .validation import selected_rates, validate


def run_tests():
    directory = ROOT/"results/preflight"
    directory.mkdir(parents=True,exist_ok=True)
    sources = source_hashes()
    manifest = directory/"tests.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (saved["source_hashes"] == sources and saved["log_sha256"] == digest(directory/"tests.log")
                and saved["failures"] == saved["errors"] == saved["skipped"] == 0):
            return saved
    suite = unittest.TestLoader().discover(str(ROOT/"tests"),top_level_dir=".")
    with (directory/"tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"All new CPU/CUDA contracts must pass: {directory/'tests.log'}")
    if source_hashes() != sources:
        raise RuntimeError("Sources changed during preflight")
    checks = {"tests":result.testsRun,"failures":0,"errors":0,"skipped":0,
              "source_hashes":sources,"log_sha256":digest(directory/"tests.log"),
              "completed_utc":datetime.now(timezone.utc).isoformat()}
    for source,target in (("/tmp/amsgrad_extensions_lean_build.log","lean_build.log"),
        ("/tmp/amsgrad_extensions_axioms.log","lean_axioms.log"),
        ("/tmp/amsgrad_extensions_individual_axioms.log","lean_individual_axioms.log")):
        shutil.copyfile(source,directory/target)
    if "Build completed successfully" not in (directory/"lean_build.log").read_text():
        raise RuntimeError("Lean modules did not build successfully")
    if "0 resting on a sorry · 0 extra axioms" not in (directory/"lean_axioms.log").read_text():
        raise RuntimeError("Lean transitive audit is not clean")
    individual = (directory/"lean_individual_axioms.log").read_text()
    if "error:" in individual or "sorryAx" in individual or individual.count("depends on axioms:") != 7:
        raise RuntimeError("Individual Lean theorem audits did not pass")
    write_json(manifest,checks)
    return checks


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing protocol changed; preserve it and choose another output: {path}")
    else:
        write_json(path,payload)


def launch(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(args.output or ROOT/"results/rtx3050")
    directory.mkdir(parents=True,exist_ok=True)
    progress = lambda message:log_progress(directory,message)
    if args.validate_only:
        print(json.dumps(validate(directory,progress=progress),indent=2),flush=True)
        write_report(directory)
        return
    tests = run_tests()
    progress(f"All {tests['tests']} new CPU/CUDA tests passed before training; Lean axiom audits passed")
    if args.tests_only:
        return
    old, common = baseline()
    config = Config(**common["config"])
    data = TextData.load("experiments/tinyshakespeare.txt","cuda")
    stop = StopConfig(**common["stopping"])
    screen_stop = StopConfig(every=250,patience=8,min_steps=250,max_steps=250)
    screen_protocol = {"source_hashes":source_hashes(),"prior_artifacts":prior_artifacts(),
        "config":asdict(config),"batch":common["batch"],"stopping":asdict(screen_stop),
        "rates":{m:list(RATES[m]) for m in METHODS},"test_evaluations":0,
        "data_sha256":data.sha256,"preflight_artifacts":preflight_artifacts()}
    freeze(directory/"screening_protocol.json",screen_protocol)
    artifacts = Path("experiments/runs/amsgrad_extensions_benchmark")/directory.name
    screen = read_results(directory/"screening.jsonl")
    completed = {r["id"] for r in screen}
    screen_jobs = [(a,m,rate) for a in common["attention"] for m in METHODS for rate in RATES[m]]
    for index,(attention,method,rate) in enumerate(screen_jobs,1):
        identifier = f"screen:{attention}:{method}:{rate:g}:0"
        if identifier in completed:
            continue
        progress(f"screen {index}/18 START {attention} {method} lr={rate:g}")
        row = train_one(config,data,attention,method,rate,0,common["batch"],screen_stop,
                        artifacts/"screen"/f"candidate{index}",progress=progress,
                        evaluate_test=False,phase="screen")
        check_initial(row,old)
        row["id"] = identifier
        with (directory/"screening.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
            stream.flush()
        screen.append(row)
        completed.add(identifier)
        progress(f"screen {index}/18 END {attention} {method} val={row['validation_loss']} status={row['status']}")
    rates = selected_rates(screen)
    write_json(directory/"selected_rates.json",rates)
    progress(f"Validation-only selected rates: {rates}")
    if args.screen_only:
        return
    groups = [(a,m) for a in common["attention"] for m in METHODS]
    random.Random(1729).shuffle(groups)
    jobs = [(a,m,s) for a,m in groups for s in common["seeds"]]
    protocol = {"config":asdict(config),"batch":common["batch"],"attention":common["attention"],
        "seeds":common["seeds"],"methods":list(METHODS),"stopping":asdict(stop),
        "selected_rates":rates,"screening_sha256":digest(directory/"screening.jsonl"),
        "data_sha256":data.sha256,"data_boundaries":list(data.boundaries),
        "characters":data.characters,"initialization":common["initialization"],
        "coefficients":{"beta1":.9,"beta2":.999,"epsilon":1e-8,"bias_correction":False,
            "w_decay":.01,"md_gain_rate":.001,"md_aux_rate":.0003,
            "guarded_md_direction_rate":.0003,"guard_sigma":.25,
            "sphere_radius":"actual initialization Frobenius norm; effective gains initially one"},
        "routing":"MD on hidden Linear matrices; AMSGrad on tied embedding and temperatures; W on all",
        "guard":"joint fused displacement, nonzero MD blocks, nonsingular per-block fallback and row repair",
        "proof_scope":"deterministic exact-real single-matrix results; GPT hypotheses unverified",
        "dtype":"float32","tf32":False,"deterministic_algorithms":True,
        "compiler":{"backend":"inductor","mode":"reduce-overhead","fullgraph":True,"dynamic":False,
            "optimizer_and_reverse_ad_compiled":True,"projection_and_guard_repair_compiled":True,
            "windows_and_evaluation_compiled":True,"timing_includes_cold_compilation":True},
        "source_hashes":source_hashes(),"prior_artifacts":prior_artifacts(),
        "preflight_artifacts":preflight_artifacts(),"selection":"exact best validation; test once after stop",
        "job_order":jobs}
    fingerprint = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    meta_path = directory/"metadata.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text())["protocol_sha256"] != fingerprint:
            raise ValueError("Existing final experiment has changed sources/settings")
    else:
        props = torch.cuda.get_device_properties(0)
        write_json(meta_path,{"protocol":protocol,"protocol_sha256":fingerprint,
            "tests":{"tests":tests["tests"],"skipped":0,"failures":0,"errors":0},
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "environment":{"gpu":props.name,"gpu_total_bytes":props.total_memory,
                "torch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version()}})
    raw = directory/"runs.jsonl"
    rows = read_results(raw)
    completed = {r["id"] for r in rows}
    for index,(attention,method,seed) in enumerate(jobs,1):
        identifier = f"{attention}:{method}:{seed}"
        if identifier in completed:
            continue
        progress(f"run {index}/18 START {identifier} lr={rates[attention][method]:g}")
        row = train_one(config,data,attention,method,rates[attention][method],seed,
            common["batch"],stop,artifacts/"final",progress=progress)
        check_initial(row,old)
        row["protocol_sha256"] = fingerprint
        with raw.open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
            stream.flush()
        rows.append(row)
        completed.add(identifier)
        write_report(directory)
        progress(f"run {index}/18 END {identifier}: {row['status']} stop={row['stop_reason']} "
                 f"updates={row['actual_steps']} best={row['best_step']} test={row['test_loss']}")
    validate(directory,progress=progress)
    write_report(directory)
    progress("COMPLETE: all18 runs, stopping replay, compiled checkpoint audits and combined rankings saved")
