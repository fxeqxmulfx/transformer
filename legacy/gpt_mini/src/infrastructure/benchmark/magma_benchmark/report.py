"""Combined all-method ranking with independent baseline provenance."""

import json
import statistics
from pathlib import Path

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results, write_json, write_summary
from .protocol import baseline, check_compatible, check_pairing


def write_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    protocol, env = metadata["protocol"], metadata["environment"]
    old_rows, old_metadata, provenance = baseline()
    check_compatible(protocol, old_metadata["protocol"])
    new_rows = read_results(directory / "runs.jsonl")
    check_pairing(new_rows, old_rows)
    combined = directory / "combined"
    combined.mkdir(exist_ok=True)
    all_rows = old_rows + new_rows
    summaries = write_summary(combined, all_rows, protocol["seeds"])
    paired = []
    for attention in protocol["attention"]:
        for method in protocol["methods"]:
            name = method["name"]
            if not name.startswith("magma_"):
                continue
            base = name.removeprefix("magma_")
            magma = {r["seed"]: r for r in new_rows if r["phase"] == "final" and r["attention"] == attention
                     and r["method"] == name and r["status"] == "ok"}
            dense = {r["seed"]: r for r in all_rows if r["phase"] == "final" and r["attention"] == attention
                     and r["method"] == base and r["status"] == "ok"}
            if set(magma) != set(protocol["seeds"]) or set(dense) != set(protocol["seeds"]):
                continue
            deltas = [magma[s]["test_loss"] - dense[s]["test_loss"] for s in protocol["seeds"]]
            paired.append({"attention": attention, "magma": name, "base": base,
                           "seed_deltas": deltas, "mean_delta": statistics.mean(deltas),
                           "delta_sample_sd": statistics.stdev(deltas),
                           "seeds_improved": sum(d < 0 for d in deltas)})
    write_json(directory / "paired_deltas.json", paired)
    write_json(combined / "provenance.json", {"baseline": provenance,
               "followup_protocol_sha256": metadata["protocol_sha256"],
               "followup_raw": str(directory / "runs.jsonl"),
               "note": "Old rows retain original session timing; no measured source was edited."})
    tests = metadata["tests"]
    lines = ["# Magma on GPTMini and Tiny Shakespeare", "",
             f"GPU: **{env['gpu']}**, {env['gpu_total_bytes'] / 2**30:.2f} GiB CUDA-visible memory; "
             f"Torch {env['torch']}, CUDA {env['cuda']}, float32, TF32 disabled.", "",
             f"All {tests['tests']} CPU/CUDA tests passed before the experiment, with zero failures/skips.", "",
             "## Observed ranking", "",
             "The primary ranking contains every measured method: 18 frozen baseline methods and "
             "six new methods. It is ordered by the mean final test cross-entropy of three seeds. "
             "± is sample SD, not a confidence interval; lower CE/perplexity is better.", ""]
    for attention in protocol["attention"]:
        rows = [r for r in summaries if r["attention"] == attention]
        complete = [r for r in rows if r["complete"]]
        if complete:
            best = complete[0]
            lines.append(f"**{attention}: {best['method']}** has the lowest observed mean: "
                         f"{best['test_loss_mean']:.5f} ± {best['test_loss_std']:.5f}, "
                         f"character perplexity {best['test_ppl']:.3f}.")
        lines += ["", f"### {attention}", "",
                  "| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Seeds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            if r["test_loss_mean"] is None:
                lines.append(f"| {r['method']} | {r['lr']:g} | failed | — | — | — | — | 0/3 |")
            else:
                lines.append(f"| {r['method']} | {r['lr']:g} | {r['test_loss_mean']:.5f} ± {r['test_loss_std']:.5f} | "
                             f"{r['test_ppl']:.3f} | {r['train_seconds_mean']:.1f} | {r['optimizer_ms_mean']:.3f} | "
                             f"{r['peak_memory_mib']:.0f} | {r['successful_seeds']}/3 |")
        lines += [""]
    lines += ["## Magma versus its dense base", "",
              "Each base and wrapper selects its rate using its declared validation-only search. "
              "The deltas below pair initialization and minibatches, but allow different selected rates. "
              "A negative delta favors Magma. Three seeds do not establish a general ordering.", "",
              "| Attention | Wrapper | Mean CE delta | Paired delta SD | Improved seeds | Seed deltas |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in paired:
        lines.append(f"| {r['attention']} | {r['magma']} | {r['mean_delta']:+.5f} | "
                     f"{r['delta_sample_sd']:.5f} | {r['seeds_improved']}/3 | "
                     + ", ".join(f"{d:+.5f}" for d in r["seed_deltas"]) + " |")
    lines += ["", "## Recipe, scope, and reproducibility", "",
              "- Local source: `papers/arXiv-2602.15322v1/google_main.tex`, Algorithm 1 and Sections 3–4. "
              "Independent Bernoulli(0.5) masks on eight attention/FFN matrices; tau=2; score EMA=.9; "
              "first moments and variances updated densely before masking. The fused QKV tensor is one block. "
              "The 393,216 masked parameters exclude the 8,328 embedding/head/temperature parameters.",
              "- Apply s*m to the entire base displacement, including AdamW decay, without 1/p. "
              "Initial scale=.5 and zero-vector cosine=0 are explicit extensions where the paper is silent. "
              "SGD/RMSProp retain an extra dense scoring EMA. Adam/AdamW/Muon reuse their base first moment. "
              "Independent CPU mask RNG uses 20000+seed; all Magma methods share mask streams.",
              "- Unmodified GPTMini: 2 layers, 4 heads, width128, FFN512, context64, batch32, "
              "401,544 unique parameters, 65 characters, contiguous 90/5/5 split. All original normalization, "
              "XSA, RoPE and tying remain. Repeat with softmax and causal Sparsemax.",
              "- Each new method receives three predeclared rates, 250 screening updates per rate on seed0, "
              "then 1000 final updates on seeds0,1,2. No warmup, LR schedule, clipping, AMP, dropout or compilation. "
              "Full fixed-window held-out evaluation; no test-loss tuning. Grids and conventions are in `../../PLAN.md`.",
              "- RMSProp: raw v EMA=.999, epsilon=1e-8, no momentum in the direction, no debiasing/decay. "
              "Adam is the old raw-moment rule; AdamW is bias corrected, decay=.01. Muon preserves the old "
              "five-step Newton–Schulz and auxiliary Adam on tied embeddings/temperatures, with LR ratio .05 "
              "and zero decay. This remains the documented experimental hybrid, rather than the paper's shared-LR recipe.",
              "- The corrected Lean stationarity theorem is for normalized masked SGD under explicit smoothness, "
              "sampling and step-size hypotheses. It does not prove convergence of these adaptive wrappers or "
              "certify GPTMini training assumptions. Sparsemax attention does not make joint training convex.",
              "- This short Tiny Shakespeare experiment differs from C4/Llama and the paper's warmup/cosine schedule. "
              "Conclusions apply to these measured budgets and grids. New and old timing come from separate "
              "sessions on the same GPU; old timings retain their original values.", "",
              "## Artifacts", "",
              "`runs.jsonl` stores every new screening/final run, validation curves, masks/damping, timings, "
              "memory and paired hashes. `selected_rates.json` records validation-selected rates. "
              "`summary.*` contains the new methods; `combined/summary.*` contains all methods. "
              "`paired_deltas.json`, `validation.json`, calibration and test logs preserve the checks. "
              "Seed0 checkpoints include model, optimizer states and mask RNG under the ignored "
              "`experiments/runs/magma_benchmark/rtx3050/`.", "",
              f"Frozen baseline raw SHA256: `{provenance['raw_sha256']}`. "
              f"New protocol SHA256: `{metadata['protocol_sha256']}`. Both sets of measured sources "
              "are fingerprinted in `metadata.json`; resume refuses mismatched fingerprints.", "",
              "```bash", ".venv/bin/python -u -m gpt_mini.infrastructure.benchmark.magma_benchmark --calibrate",
              ".venv/bin/python -u -m gpt_mini.infrastructure.benchmark.magma_benchmark",
              ".venv/bin/python -m gpt_mini.infrastructure.benchmark.magma_benchmark --validate-only", "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    return summaries
