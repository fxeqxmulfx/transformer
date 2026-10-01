"""Keep validation-stopped ranking separate from fixed-budget measurements."""

import json
from pathlib import Path

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results
from .storage import write_summary


def write_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    protocol, env = metadata["protocol"], metadata["environment"]
    rows = read_results(directory / "runs.jsonl")
    summaries = write_summary(directory, rows, protocol["seeds"])
    stop = protocol["stopping"]
    lines = ["# GPTMini comparison with validation patience", "",
             f"GPU: **{env['gpu']}**, Torch {env['torch']}, float32, TF32 disabled. "
             f"{metadata['tests']['tests']} CPU/CUDA tests passed before training.", "",
             f"Completed {len(rows)}/{len(protocol['methods']) * 2 * len(protocol['seeds'])} runs. "
             "Rank by test CE of the best validation checkpoint; lower is better. "
             "± is sample SD over seeds, not a confidence interval. Incomplete groups remain visible.", "",
             "## Shared stopping rule", "",
             f"- Full validation check every {stop['every']} updates; patience {stop['patience']} checks "
             f"without a decrease larger than {stop['min_delta']} CE from the improvement anchor.",
             f"- Stop after {stop['divergence_patience']} consecutive checks at least {stop['divergence_delta']} CE "
             f"above the best validation value. Ordinary stopping begins at {stop['min_steps']} updates.",
             f"- Stop nonfinite losses immediately. Emergency cap: {stop['max_steps']} updates. "
             "`max_steps` denotes a censored budget, not plateau/convergence.",
             "- Save every exact validation minimum, even below min_delta; restore that checkpoint. "
             "Test is evaluated once, after stopping. Recovered unstable runs are counted and identified.",
             "- Identical model, character split, initialization, minibatch prefixes, optimizer recipes and "
             "previous validation-selected learning rates. No LR retuning, schedule, warmup, AMP or clipping. "
             "Early-stopping quality and realized compute budget are both part of the comparison.", ""]
    for attention in protocol["attention"]:
        group = [r for r in summaries if r["attention"] == attention]
        complete = [r for r in group if r["complete"]]
        lines += [f"## {attention}", ""]
        if complete:
            best = complete[0]
            lines += [f"Lowest complete mean: **{best['method']}**, {best['test_loss_mean']:.5f} "
                      f"± {best['test_loss_std']:.5f}, PPL {best['test_ppl']:.3f}.", ""]
        lines += ["| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in group:
            if r["test_loss_mean"] is None:
                lines.append(f"| {r['method']} | {r['lr']:g} | failed | — | — | — | — | — | "
                             f"{r['capped_runs']} | {r['recovered_runs']} | 0/{r['expected_seeds']} |")
            else:
                lines.append(f"| {r['method']} | {r['lr']:g} | {r['test_loss_mean']:.5f} ± {r['test_loss_std']:.5f} | "
                             f"{r['test_ppl']:.3f} | {r['best_step_mean']:.0f} | {r['actual_steps_mean']:.0f} | "
                             f"{r['train_seconds_mean']:.1f} | {r['total_seconds_mean']:.1f} | "
                             f"{r['capped_runs']} | {r['recovered_runs']} | {r['usable_seeds']}/{r['expected_seeds']} |")
        lines += [""]
    lines += ["## Scope and artifacts", "",
              "This ranking contains only this stopping protocol's runs, including all measured methods "
              "regardless of proof status. It is not combined with the earlier 1000-update last-iterate ranking. "
              "Those frozen sources/results remain intact, and their fingerprints are in metadata. "
              "Muon retains its recorded auxiliary Adam recipe; Magma retains Algorithm 1 with dense moments, "
              "p=.5, tau=2 and no 1/p. Convergence theorem assumptions are not certified by this experiment.", "",
              "`runs.jsonl`: completed runs, all validation curves, realized budgets, reasons, hashes, memory and timing. "
              "`summary.json`/`summary.csv`: rankings. `metadata.json`: stopping settings, environment, data/source hashes "
              "and tests. `validation.json`: replayed decisions and CUDA checkpoint re-evaluation. "
              "Best model checkpoints and live curves are under ignored `experiments/runs/patience_benchmark/`. "
              "Resume restarts an unfinished run and skips completed identifiers under the same fingerprint.", "",
              "```bash", ".venv/bin/python -u -m gpt_mini.infrastructure.benchmark.patience_benchmark --all",
              ".venv/bin/python -m gpt_mini.infrastructure.benchmark.patience_benchmark --all --validate-only", "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
