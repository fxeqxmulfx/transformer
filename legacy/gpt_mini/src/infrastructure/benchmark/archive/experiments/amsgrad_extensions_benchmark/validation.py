"""Replay selection/stopping, frozen fingerprints and fresh CUDA checkpoints."""

import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model
from experiments.optimizer_benchmark.data import TextData, training_starts
from experiments.optimizer_benchmark.runner import state_hash
from experiments.optimizer_benchmark.storage import read_results, write_json
from experiments.patience_benchmark.stopping import EarlyStopper, StopConfig, choose_best_step
from experiments.compiled_benchmark.telemetry import snapshot
from experiments.full_compile_benchmark.storage import summarize
from .protocol import baseline, check_initial, digest, preflight_artifacts, prior_artifacts, source_hashes
from .step import ExtensionStep


def selected_rates(screening):
    chosen = {}
    for attention in ("softmax","sparsemax"):
        chosen[attention] = {}
        for method in ("amsgradw","amsgradmd","amsgradmd_guarded"):
            candidates = [r for r in screening if r["attention"] == attention and r["method"] == method]
            if len(candidates) != 3 or any(r["test_evaluations"] != 0 or r["test_loss"] is not None for r in candidates):
                raise ValueError("Screen must use three rates and no test evaluation")
            valid = [r for r in candidates if r["status"] == "ok" and math.isfinite(r["validation_loss"])]
            chosen[attention][method] = min(valid,key=lambda r:(r["validation_loss"],r["lr"]))["lr"]
    return chosen


def replay_stopping(row, stop):
    controller = EarlyStopper(stop)
    last = None
    for curve in row["curves"]:
        if last is not None and last.should_stop:
            raise ValueError("Updates continued beyond a stop decision")
        value = curve["validation_loss"] if curve["validation_loss"] is not None else math.nan
        last = controller.observe(curve["step"],value)
        if (last.new_best != curve["new_best"] or last.reason != curve["stop_reason"]
                or controller.bad_checks != curve["bad_checks"]
                or controller.divergent_checks != curve["divergent_checks"]):
            raise ValueError("Stopping replay differs")
    if row["stop_reason"] in {"patience","validation_divergence","max_steps","nonfinite_validation"}:
        if not last.should_stop or last.reason != row["stop_reason"]:
            raise ValueError("Run has no matching stopping decision")
    if row["best_step"] != choose_best_step(row["curves"]):
        raise ValueError("Best checkpoint was not selected by validation")


def check_checkpoint(row, config, data, batch):
    torch.compiler.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    payload = torch.load(row["checkpoint"],map_location="cpu",weights_only=True)
    model = make_model(config,row["attention"],row["seed"],"cuda")
    runtime = ExtensionStep(model,row["attention"],row["method"],row["lr"],row["seed"],batch,1)
    try:
        model.load_state_dict(payload["model"])
        if state_hash(model) != row["best_model_sha256"]:
            raise ValueError("Selected model hash changed")
        val = runtime.evaluate(data.validation,batch)
        test = runtime.evaluate(data.test,batch)
        if abs(val-row["validation_loss"]) > 1e-9 or abs(test-row["test_loss"]) > 1e-9:
            raise ValueError("Checkpoint does not reproduce held-out losses")
        # Repeated validation includes warmup and recording of the reset case.
        runtime.evaluate(data.validation,batch)
        state = snapshot()
        if state["graph_breaks"] or state["recorded_graph_nodes"] <= 0:
            raise ValueError("Checkpoint evaluation lacks compiled CUDA Graphs")
        return {"id":row["id"],"validation_loss":val,"test_loss":test,**state}
    finally:
        runtime.close()
        del runtime,model,payload
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()


def validate(directory, *, complete=True, progress=None):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    p = metadata["protocol"]
    fingerprint = hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    if (fingerprint != metadata["protocol_sha256"] or p["source_hashes"] != source_hashes()
            or p["prior_artifacts"] != prior_artifacts() or p["preflight_artifacts"] != preflight_artifacts()):
        raise ValueError("Protocol, sources or previous measurements changed")
    if digest(directory/"screening.jsonl") != p["screening_sha256"]:
        raise ValueError("Validation rate selection changed")
    rates = selected_rates(read_results(directory/"screening.jsonl"))
    if rates != p["selected_rates"]:
        raise ValueError("Recorded rates differ from validation-only selection")
    earlier, _ = baseline()
    config, stop = Config(**p["config"]), StopConfig(**p["stopping"])
    data = TextData.load("experiments/tinyshakespeare.txt","cuda")
    if data.sha256 != p["data_sha256"]:
        raise ValueError("Dataset changed")
    rows = read_results(directory/"runs.jsonl")
    expected = {f"{a}:{m}:{s}" for a in p["attention"] for m in p["methods"] for s in p["seeds"]}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected or complete and set(ids) != expected:
        raise ValueError("Duplicate, unknown or missing runs")
    checks = []
    for row in rows:
        check_initial(row,earlier)
        if row["protocol_sha256"] != fingerprint or row["lr"] != rates[row["attention"]][row["method"]]:
            raise ValueError("Changed per-run settings")
        starts = training_starts(len(data.train),config.max_seq_len,p["batch"],stop.max_steps,row["seed"])
        for name,plan in (("batch_plan_sha256",starts),("batch_sha256",starts[:row["actual_steps"]])):
            if hashlib.sha256(plan.numpy().tobytes()).hexdigest() != row[name]:
                raise ValueError("Unpaired minibatch stream")
        replay_stopping(row,stop)
        if row["test_evaluations"] != 1:
            raise ValueError("Final test was not evaluated exactly once after stopping")
        if row["status"] == "failed":
            continue
        if digest(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        state = row["compile"]["final"]
        if state["graph_breaks"] or state["recorded_graph_nodes"] <= 0:
            raise ValueError("Training lacks complete compiled CUDA Graphs")
        evaluations = row["compile"]["evaluations"]
        warm = evaluations[2] if len(evaluations) >= 3 else evaluations[-1]
        keys = ("unique_fx_graphs","traced_calls","graph_breaks","recorded_graph_nodes")
        if any(any(e[k] != warm[k] for k in keys) for e in evaluations[3:]+[state]):
            raise ValueError("Repeated tracing/recording after mode warmup")
        checks.append(check_checkpoint(row,config,data,p["batch"]))
        if progress:
            progress(f"checkpoint audit {len(checks)}/{len(rows)}: {row['id']}")
    summaries = json.loads((directory/"summary.json").read_text())["methods"]
    if summaries != summarize(rows,p["seeds"]):
        raise ValueError("Summary differs from raw measurements")
    result = {"runs":len(rows),"expected_runs":len(expected),"complete":set(ids)==expected,
        "failed":sum(r["status"]=="failed" for r in rows),"recovered":sum(r["status"]=="recovered" for r in rows),
        "stopping_replayed":True,"screening_test_evaluations":0,
        "initialization_and_minibatches_paired":True,"previous_experiments_unchanged":True,
        "steady_compiler_counters":True,"compiled_cuda_checkpoint_checks":checks,
        "raw_sha256":digest(directory/"runs.jsonl")}
    write_json(directory/"validation.json",result)
    return result
