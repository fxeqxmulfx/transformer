"""Rank every measured recipe, with new variants and proof scope visible."""

from pathlib import Path

from experiments.full_compile_benchmark.storage import summarize, write_summary
from experiments.optimizer_benchmark.storage import read_results, write_json
from .protocol import BASELINE, METHODS


def write_report(directory):
    directory = Path(directory)
    if not (directory / "runs.jsonl").exists():
        return
    rows = read_results(directory / "runs.jsonl")
    seeds = [0,1,2]
    write_summary(directory,rows,seeds)
    old = read_results(BASELINE / "runs.jsonl")
    combined = summarize(old+rows,seeds)
    write_json(directory / "combined_summary.json",{"methods":combined})
    lines = ["# AMSGradW and AMSGradMD on GPTMini", "",
        f"Completed new runs: {len(rows)}/18. All24 baseline: 144/144.", "",
        "The table uses the exact best validation checkpoint of each run.",
        "Test CE is mean +/- sample SD over three seeds; lower is better.",
        "The main ranking includes every usable method, including original MD.", "",
        "Float32, RTX3050, full compiled steps and CUDA Graphs; the original",
        "paired data/models/patience settings are preserved. Training seconds",
        "include cold compilation and exclude validation/checkpoint time.", "",
        "## New variants and the existing AMSGrad reference", "",
        "| Attention | Method | Test CE mean +/- SD | Rate | Updates mean | Train seconds mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in combined:
        if r["method"] not in METHODS+("amsgrad",) or r["test_loss_mean"] is None:
            continue
        partial = " (partial)" if not r["complete"] else ""
        lines.append(f"| {r['attention']} | {r['method']}{partial} | "
            f"{r['test_loss_mean']:.6f} +/- {r['test_loss_std']:.6f} | {r['lr']:g} | "
            f"{r['actual_steps_mean']:.0f} | {r['train_seconds_mean']:.1f} |")
    lines += ["", "Rates are: the W rate, raw MD direction rate, and guarded MD",
        "fallback tau, respectively. MD gain rate is0.001; auxiliary rate is0.0003.",
        "The guarded proposal itself uses direction rate0.0003. See PLAN.md.", ""]
    guards = [r for r in rows if r["method"].endswith("guarded") and r["guard_acceptance"] is not None]
    if guards:
        lines += ["## Guard behavior", "",
            "| Attention | Seed | Accepted proposals | Acceptance fraction | Halved zero steps |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for r in guards:
            d = r["optimizer_diagnostics"]
            lines.append(f"| {r['attention']} | {r['seed']} | {d['accepted_steps']} | "
                         f"{r['guard_acceptance']:.6f} | {d['halved_matrix_steps']} |")
    for attention in ("softmax","sparsemax"):
        lines += ["",f"## All measured methods: {attention}","",
            "| Rank | Method | Seeds | Test CE mean +/- SD | Stop reasons |",
            "| ---: | --- | ---: | ---: | --- |"]
        ranked = [r for r in combined if r["attention"] == attention and r["test_loss_mean"] is not None]
        for rank,r in enumerate(ranked,1):
            partial = " (partial)" if not r["complete"] else ""
            lines.append(f"| {rank} | {r['method']}{partial} | {r['usable_seeds']} | "
                         f"{r['test_loss_mean']:.6f} +/- {r['test_loss_std']:.6f} | {r['stop_reasons']} |")
    lines += ["", "## Lean proof scope", "",
        "AMSGradW convergence needs L < decay*epsilon and gives a history-dependent",
        "weighted regularized equilibrium, generally not zero original gradient.",
        "Original AMSGradMD has a proved counterexample. Guarded MD has deterministic",
        "stationarity/loss convergence; strong convexity adds weight convergence.",
        "These hypotheses are not established for minibatch GPTMini. The Python",
        "multi-block guard is a documented extension of the single-matrix Lean model.",
        "All proposal/gain/auxiliary buffers use AMSGrad; no Adam is hidden in routing.",
        "Three seeds and the rate screen do not establish a general superiority.",
        "See LEAN_AUDIT.md, metadata.json, screening.jsonl and validation.json.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
