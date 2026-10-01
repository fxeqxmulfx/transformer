# Compiled GPTMini comparison with validation patience

GPU: **NVIDIA GeForce RTX 3050 Laptop GPU**, Torch 2.14.0+cu130, float32, TF32 disabled. 80 CPU/CUDA tests passed before training.

Completed 6/144 runs. Rank by test CE of the best validation checkpoint; lower is better. ± is sample SD over seeds, not a confidence interval. Incomplete groups remain visible.

## Shared stopping rule

- Full validation check every 250 updates; patience 8 checks without a decrease larger than 0.0001 CE from the improvement anchor.
- Stop after 3 consecutive checks at least 0.1 CE above the best validation value. Ordinary stopping begins at 1000 updates.
- Stop nonfinite losses immediately. Emergency cap: 20000 updates. `max_steps` denotes a censored budget, not plateau/convergence.
- Save every exact validation minimum, even below min_delta; restore that checkpoint. Test is evaluated once, after stopping. Recovered unstable runs are counted and identified.
- Identical model, character split, initialization, minibatch prefixes, optimizer recipes and previous validation-selected learning rates. No LR retuning, schedule, warmup, AMP or clipping. Early-stopping quality and realized compute budget are both part of the comparison.

## softmax

Lowest complete mean: **adafisher**, 1.65176 ± 0.00944, PPL 5.216.

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adafisher | 0.001 | 1.65176 ± 0.00944 | 5.216 | 19500 | 20000 | 247.3 | 252.0 | 3 | 0 | 3/3 |

## sparsemax

Lowest complete mean: **adagrad**, 1.69125 ± 0.02762, PPL 5.426.

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adagrad | 0.03 | 1.69125 ± 0.02762 | 5.426 | 16000 | 17333 | 83.9 | 87.8 | 1 | 0 | 3/3 |

## Compiler and CUDA Graph evidence

Inductor, `torch.compile(mode="reduce-overhead", dynamic=False)` applied in place. Full model compilation for all methods except AdaFisher/AdaFisherW, whose current-activation and derivative hooks require partial compilation. Optimizer steps are eager. Sparsemax uses the tested capture adapter with unchanged finite-row mathematics.

The runtime node counter checks actual CUDAGraph objects. On each of six representative attention/method cases, the 1220-step telemetry test found no new FX traces after training and the two validation shapes were compiled, and no new CUDA graph records in three subsequent train/validation cycles. AdaFisher has more graph segments from hooks. Each measured run also stores compiler snapshots after every evaluation.

The warning `Not enough SMs to use max_autotune_gemm mode` occurred during preflight. No CUDA Graph execution error occurred in those tests or timing probes. This is preflight evidence, not a guarantee about every long optimizer trajectory.

Training/total seconds in the ranking include cold compilation. `compile.cold_forward_seconds` separates first forward costs; the timing probe below uses 20 warmup steps and 200 timed updates on the actual model and batch.

| Attention | Method | Eager ms/update | Compiled ms/update | Speedup |
| --- | --- | ---: | ---: | ---: |
| softmax | adamw | 9.455 | 4.669 | 2.02x |
| softmax | magma_muon | 14.515 | 9.914 | 1.46x |
| softmax | adafisher | 12.350 | 11.690 | 1.06x |
| sparsemax | adamw | 9.958 | 4.912 | 2.03x |
| sparsemax | magma_muon | 16.808 | 11.600 | 1.45x |
| sparsemax | adafisher | 20.076 | 13.890 | 1.45x |

The compiled ranking uses its own protocol and retains all earlier fixed-budget and partial eager results separately.

## Scope and artifacts

This ranking contains only this stopping protocol's runs, including all measured methods regardless of proof status. It is not combined with the earlier 1000-update last-iterate ranking. Those frozen sources/results remain intact, and their fingerprints are in metadata. Muon retains its recorded auxiliary Adam recipe; Magma retains Algorithm 1 with dense moments, p=.5, tau=2 and no 1/p. Convergence theorem assumptions are not certified by this experiment.

`runs.jsonl`: completed runs, all validation curves, realized budgets, reasons, hashes, memory and timing. `summary.json`/`summary.csv`: rankings. `metadata.json`: stopping settings, environment, data/source hashes and tests. `validation.json`: replayed decisions and CUDA checkpoint re-evaluation. Best model checkpoints and live curves are under ignored `experiments/runs/compiled_benchmark/`. Resume restarts an unfinished run and skips completed identifiers under the same fingerprint.

```bash
.venv/bin/python -u -m experiments.compiled_benchmark --all
.venv/bin/python -m experiments.compiled_benchmark --all --validate-only
```
