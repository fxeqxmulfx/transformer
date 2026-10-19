"""Write the final experiment report directly from raw measurements."""

import json
from pathlib import Path

from .registry import METHODS
from .storage import read_results, write_summary, winners


def write_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    protocol = metadata["protocol"]
    results = read_results(directory / "runs.jsonl")
    summaries = write_summary(directory, results, protocol["seeds"])
    best = winners(summaries)
    environment = metadata["environment"]
    selected = json.loads((directory / "selected_rates.json").read_text())
    lines = ["# GPTMini optimizer comparison on Tiny Shakespeare", "",
             f"Device: **{environment['gpu']}**, {environment['gpu_total_bytes'] / 2**30:.2f} GiB CUDA-visible memory.",
             f"PyTorch {environment['torch']}; CUDA runtime {environment['cuda']}; float32, TF32 disabled.", "",
             "## Observed winners", ""]
    for attention in protocol["attention"]:
        winner = best.get(attention)
        if winner:
            lines.append(f"- **{attention}: {winner['method']}**, final test cross-entropy "
                         f"{winner['test_loss_mean']:.5f} ± {winner['test_loss_std']:.5f}, "
                         f"character perplexity {winner['test_ppl']:.3f}, "
                         f"{winner['train_seconds_mean']:.1f} seconds per final run.")
    lines += ["", "The ± value is sample standard deviation across seeds, not a confidence interval. "
              "These are the lowest observed three-seed means in this experiment; close scores "
              "do not establish a reliable or general optimizer ordering.", "", "## Protocol", "",
              f"- Reference: unmodified `experiments/gpt_mini.py`; model configuration: `{protocol['config']}`.",
              f"- Character tokenizer: {len(protocol['characters'])} characters. Contiguous 90/5/5 split; "
              f"boundaries `{protocol['data_boundaries']}`. Dataset SHA256: `{protocol['data_sha256']}`.",
              f"- Batch {protocol['batch']}; unique matrix initialization Normal(0, 0.02), with the "
              "source temperatures and embedding/unembedding tying preserved.",
              f"- Each method and attention type receives three declared learning rates, "
              f"{protocol['screen_steps']} updates per rate on seed 0, selected only by final validation loss.",
              f"- Final runs: {protocol['final_steps']} updates each, seeds `{protocol['seeds']}`. "
              "Same initial parameters and minibatch plans across methods and attention types.",
              "- Held-out losses evaluate all complete nonoverlapping context windows. Test data is "
              "evaluated only in final runs, after learning-rate selection.",
              "- No AMP, gradient clipping, dropout, early stopping or compilation. Report last-iterate loss. "
              "Training time excludes held-out evaluation; optimizer time is measured with CUDA events.",
              "- Sparsemax is the causal Euclidean projection of the unchanged QK/RoPE score row "
              "onto the simplex, matching `Transformer.GPTMini.Convex.Attention`. XSA and FFN remain unchanged.",
              f"- Tests passed before GPU training: {metadata['tests']['tests']}; no failures or skips. "
              "See `tests.log`. Source hashes, environment and protocol hash are in `metadata.json`.", ""]
    for attention in protocol["attention"]:
        lines += [f"## {attention} results", "",
                  "| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Guard accepted | Seeds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in [s for s in summaries if s["attention"] == attention]:
            if row["test_loss_mean"] is None:
                lines.append(f"| {row['method']} | {row['lr']:g} | failed | — | — | — | — | — | 0/{row['expected_seeds']} |")
                continue
            accepted = f"{100 * row['guard_acceptance_mean']:.1f}%" if row["guard_acceptance_mean"] is not None else "—"
            lines.append(f"| {row['method']} | {row['lr']:g} | "
                         f"{row['test_loss_mean']:.5f} ± {row['test_loss_std']:.5f} | {row['test_ppl']:.3f} | "
                         f"{row['train_seconds_mean']:.1f} | {row['optimizer_ms_mean']:.3f} | "
                         f"{row['peak_memory_mib']:.0f} | {accepted} | "
                         f"{row['successful_seeds']}/{row['expected_seeds']} |")
        lines.append("")
        viable = [s for s in summaries if s["attention"] == attention and s["complete"]]
        if viable:
            fastest = min(viable, key=lambda s: s["train_seconds_mean"])
            lines += [f"Fastest completed method: **{fastest['method']}**, "
                      f"{fastest['train_seconds_mean']:.1f} s for the same final update budget.", ""]
        missing = [name for name, rate in selected[attention].items() if rate is None]
        if missing:
            lines += [f"All screened rates failed for: `{missing}`.", ""]
    lines += ["## Implementation and proof scope", "",
              "The local convergence theorems have explicit assumptions about fixed objectives, "
              "smoothness, step bounds or convex projected online problems. Stochastic minibatches "
              "and a nonconvex GPT training loss are an empirical test, not a verification of those "
              "hypotheses. A sparse convex attention-row projection does not make joint training convex.", "",
              "| Method | Local formalization / role |", "| --- | --- |"]
    included = {m["name"] for m in protocol["methods"]}
    lines += [f"| {method.name} | {method.scope} |" for method in METHODS if method.name in included]
    lines += ["",
              "- AMSGrad/AdamX/AdamNC retain paper raw moments without Adam debiasing. Numerical "
              "epsilon is 1e-8. Inverse momentum is 0.9/t; geometric momentum is 0.9·0.99^(t−1); "
              "their step is base LR/sqrt(t). AdamNC second moments are the actual mean of past g². "
              "Constant AMSGrad/Adam use β1=0.9, β2=0.999. AdamW uses bias correction and decay 0.01.",
              "- Muon: zero-initialized raw momentum 0.95·M+G, Nesterov input 0.95·M_new+G, "
              "five printed Newton–Schulz steps in float32, and shape scale 0.2·sqrt(max(m,n)). "
              "The tied embedding and temperatures use bias-corrected Adam with 0.05 times the Muon LR; decay is zero.",
              "- DASH: block size 32 including residual blocks; left/right EMA β=0.99; damping 1e-4. "
              "Blockwise Adam grafting uses raw β1=0.9, β2=0.999 and no debiasing. NDB chains "
              "two actual six-step solves; CN uses eight fourth-root steps; both have checked "
              "Rayleigh scaling with a Frobenius/row certificate and output rescaling. Chebyshev "
              "uses degree 60, 1,000 cosine nodes, sample repair and corrected original-scale regularization/output.",
              "- The Muon/DASH training guard is the literal joint-parameter alignment/length "
              "check with σ=0.5. A rejected candidate becomes the current minibatch gradient, "
              "while all optimizer histories are retained. Acceptance counts expose this behavior. "
              "No smoothness constant is certified for this stochastic experiment.",
              "- AdaFisher: genuine Linear-input and Linear-output-backpropagation squared sums; "
              "fresh-factor EMA γ=0.8; separate min/max normalization with the proved constant-range "
              "zero extension; damping 0.001; β1=0.9 and positive-time bias correction. "
              "The tied embedding uses its unembedding Linear factors and combined parameter gradient; "
              "temperatures use the identity-factor fallback. Shared-weight curvature is a modeling "
              "approximation. AdaFisherW adds decoupled decay 0.01; other AdaFisher decay is zero.", "",
              "## Artifacts", "",
              "- `runs.jsonl`: every screening/final run, failures, loss curves, hashes and timing.",
              "- `selected_rates.json`: separate validation-selected rates for each attention type.",
              "- `summary.csv`, `summary.json`: all final three-seed measurements and observed winners.",
              "- `metadata.json`, `tests.log`, `progress.log`: environment, exact source/data fingerprints, tests and progress.",
              "- Seed-0 final checkpoints: `experiments/runs/optimizer_benchmark/rtx3050/` (gitignored).", "",
              "Reproduce from the repository root:", "", "```bash",
              ".venv/bin/python -m experiments.compare_optimizers", "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    return best
