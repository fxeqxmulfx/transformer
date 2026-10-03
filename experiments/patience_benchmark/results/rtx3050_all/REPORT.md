# GPTMini comparison with validation patience

GPU: **NVIDIA GeForce RTX 3050 Laptop GPU**, Torch 2.14.0+cu130, float32, TF32 disabled. 74 CPU/CUDA tests passed before training.

Completed 1/144 runs. Rank by test CE of the best validation checkpoint; lower is better. ± is sample SD over seeds, not a confidence interval. Incomplete groups remain visible.

## Shared stopping rule

- Full validation check every 250 updates; patience 8 checks without a decrease larger than 0.0001 CE from the improvement anchor.
- Stop after 3 consecutive checks at least 0.1 CE above the best validation value. Ordinary stopping begins at 1000 updates.
- Stop nonfinite losses immediately. Emergency cap: 20000 updates. `max_steps` denotes a censored budget, not plateau/convergence.
- Save every exact validation minimum, even below min_delta; restore that checkpoint. Test is evaluated once, after stopping. Recovered unstable runs are counted and identified.
- Identical model, character split, initialization, minibatch prefixes, optimizer recipes and previous validation-selected learning rates. No LR retuning, schedule, warmup, AMP or clipping. Early-stopping quality and realized compute budget are both part of the comparison.

## softmax

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## sparsemax

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adagrad | 0.03 | 1.66691 ± 0.00000 | 5.296 | 14000 | 16000 | 159.9 | 166.5 | 0 | 0 | 1/3 |

## Scope and artifacts

This ranking contains only this stopping protocol's runs, including all measured methods regardless of proof status. It is not combined with the earlier 1000-update last-iterate ranking. Those frozen sources/results remain intact, and their fingerprints are in metadata. Muon retains its recorded auxiliary Adam recipe; Magma retains Algorithm 1 with dense moments, p=.5, tau=2 and no 1/p. Convergence theorem assumptions are not certified by this experiment.

`runs.jsonl`: completed runs, all validation curves, realized budgets, reasons, hashes, memory and timing. `summary.json`/`summary.csv`: rankings. `metadata.json`: stopping settings, environment, data/source hashes and tests. `validation.json`: replayed decisions and CUDA checkpoint re-evaluation. Best model checkpoints and live curves are under ignored `experiments/runs/patience_benchmark/`. Resume restarts an unfinished run and skips completed identifiers under the same fingerprint.

```bash
.venv/bin/python -u -m experiments.patience_benchmark --all
.venv/bin/python -m experiments.patience_benchmark --all --validate-only
```
