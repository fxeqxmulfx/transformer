# Full-step compiled GPTMini comparison with validation patience

GPU: **NVIDIA GeForce RTX 3050 Laptop GPU**, Torch 2.14.0+cu130, float32, TF32 disabled. 85 CPU/CUDA tests passed before training.

The entire GPU training step uses Inductor with `fullgraph=True`: input windows, GPTMini forward, cross-entropy, functional reverse AD, all dense optimizer histories, DASH roots/grafting, Muon auxiliary Adam, guards, decay and Magma masking/scales. Evaluation window gathering, loss reduction and accumulation are compiled too. AdaFisher uses zero Linear-output probes for genuine output derivatives, checked against hooks. Host logging, serialization and stopping decisions are Python orchestration. Whole-step timing includes cold compilation; optimizer time is not separable inside the full graph.

Completed 144/144 runs. Rank by test CE of the best validation checkpoint; lower is better. ± is sample SD over seeds, not a confidence interval. Incomplete groups remain visible.

## Shared stopping rule

- Full validation check every 250 updates; patience 8 checks without a decrease larger than 0.0001 CE from the improvement anchor.
- Stop after 3 consecutive checks at least 0.1 CE above the best validation value. Ordinary stopping begins at 1000 updates.
- Stop nonfinite losses immediately. Emergency cap: 20000 updates. `max_steps` denotes a censored budget, not plateau/convergence.
- Save every exact validation minimum, even below min_delta; restore that checkpoint. Test is evaluated once, after stopping. Recovered unstable runs are counted and identified.
- Identical model, character split, initialization, minibatch prefixes, optimizer recipes and previous validation-selected learning rates. No LR retuning, schedule, warmup, AMP or clipping. Early-stopping quality and realized compute budget are both part of the comparison.

## softmax

Lowest complete mean: **amsgrad**, 1.62787 ± 0.00075, PPL 5.093.

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amsgrad | 0.0003 | 1.62787 ± 0.00075 | 5.093 | 12333 | 14333 | 61.3 | 64.6 | 0 | 0 | 3/3 |
| adam | 0.0003 | 1.62984 ± 0.00316 | 5.103 | 12500 | 14500 | 62.3 | 65.8 | 0 | 0 | 3/3 |
| adamnc | 0.01 | 1.63017 ± 0.00628 | 5.105 | 13833 | 15833 | 68.8 | 72.4 | 0 | 0 | 3/3 |
| rmsprop | 0.0001 | 1.63226 ± 0.00917 | 5.115 | 16333 | 17833 | 76.9 | 80.9 | 0 | 0 | 3/3 |
| dash_ndb | 0.001 | 1.64119 ± 0.00455 | 5.161 | 11667 | 13417 | 214.1 | 217.7 | 0 | 0 | 3/3 |
| dash_cn | 0.001 | 1.64308 ± 0.00988 | 5.171 | 11583 | 13500 | 216.7 | 220.2 | 0 | 0 | 3/3 |
| dash_evd | 0.001 | 1.64338 ± 0.00960 | 5.173 | 11583 | 13500 | 222.5 | 225.7 | 0 | 0 | 3/3 |
| dash_chebyshev | 0.001 | 1.64374 ± 0.01055 | 5.175 | 11583 | 13583 | 377.6 | 381.0 | 0 | 0 | 3/3 |
| dash_ndb_guarded | 0.1 | 1.64412 ± 0.00772 | 5.176 | 11583 | 13583 | 215.7 | 219.0 | 0 | 0 | 3/3 |
| sgd | 0.1 | 1.64412 ± 0.00772 | 5.176 | 11583 | 13583 | 57.7 | 61.0 | 0 | 0 | 3/3 |
| muon_guarded | 0.1 | 1.64412 ± 0.00772 | 5.176 | 11583 | 13583 | 80.1 | 83.4 | 0 | 0 | 3/3 |
| adafisherw | 0.001 | 1.64915 ± 0.00635 | 5.203 | 18250 | 19083 | 94.3 | 98.7 | 2 | 0 | 3/3 |
| muon | 0.03 | 1.65085 ± 0.00479 | 5.211 | 8500 | 10500 | 61.1 | 63.9 | 0 | 0 | 3/3 |
| adamw | 0.001 | 1.65183 ± 0.01315 | 5.217 | 11167 | 13167 | 61.2 | 64.4 | 0 | 0 | 3/3 |
| magma_adam | 0.0009 | 1.65397 ± 0.00782 | 5.228 | 9333 | 11333 | 51.9 | 54.7 | 0 | 0 | 3/3 |
| adafisher | 0.001 | 1.66009 ± 0.00978 | 5.260 | 18000 | 19000 | 92.2 | 98.5 | 2 | 0 | 3/3 |
| magma_rmsprop | 0.0003 | 1.66104 ± 0.01321 | 5.265 | 14917 | 16500 | 73.5 | 77.2 | 1 | 0 | 3/3 |
| adagrad | 0.03 | 1.66639 ± 0.01295 | 5.293 | 13833 | 15833 | 66.6 | 70.2 | 0 | 0 | 3/3 |
| magma_muon | 0.03 | 1.66963 ± 0.01369 | 5.310 | 8583 | 10583 | 65.3 | 68.2 | 0 | 0 | 3/3 |
| magma_adamw | 0.003 | 1.67626 ± 0.01041 | 5.346 | 9000 | 11000 | 56.5 | 59.3 | 0 | 0 | 3/3 |
| magma_sgd | 0.3 | 1.68731 ± 0.00310 | 5.405 | 15833 | 17250 | 77.6 | 81.5 | 1 | 0 | 3/3 |
| adamx | 0.003 | 1.70390 ± 0.00172 | 5.495 | 19750 | 20000 | 85.4 | 89.6 | 3 | 0 | 3/3 |
| amsgrad_geometric | 0.003 | 1.76392 ± 0.01832 | 5.835 | 19917 | 20000 | 88.4 | 92.7 | 3 | 0 | 3/3 |
| amsgrad_inverse | 0.003 | 1.84751 ± 0.05307 | 6.344 | 20000 | 20000 | 84.5 | 88.8 | 3 | 0 | 3/3 |

## sparsemax

Lowest complete mean: **amsgrad**, 1.64600 ± 0.01707, PPL 5.186.

| Method | LR | Best-checkpoint test CE ± SD | PPL | Mean best step | Mean stop step | Train s | Total s | Capped | Recovered | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amsgrad | 0.0003 | 1.64600 ± 0.01707 | 5.186 | 10917 | 12917 | 57.9 | 61.3 | 0 | 0 | 3/3 |
| adamnc | 0.01 | 1.64794 ± 0.01194 | 5.196 | 14667 | 16667 | 73.8 | 77.9 | 0 | 0 | 3/3 |
| dash_evd | 0.001 | 1.65553 ± 0.00788 | 5.236 | 11583 | 13583 | 227.0 | 230.5 | 0 | 0 | 3/3 |
| dash_chebyshev | 0.001 | 1.65813 ± 0.01707 | 5.250 | 13083 | 15083 | 422.4 | 426.4 | 0 | 0 | 3/3 |
| adafisherw | 0.001 | 1.65838 ± 0.00842 | 5.251 | 17500 | 19000 | 96.2 | 100.6 | 1 | 0 | 3/3 |
| adam | 0.0003 | 1.65910 ± 0.01508 | 5.255 | 10000 | 12000 | 54.8 | 58.1 | 0 | 0 | 3/3 |
| rmsprop | 0.0003 | 1.65934 ± 0.01667 | 5.256 | 11583 | 13583 | 61.3 | 64.9 | 0 | 0 | 3/3 |
| magma_rmsprop | 0.0003 | 1.65944 ± 0.02065 | 5.256 | 15250 | 17250 | 79.8 | 84.0 | 0 | 0 | 3/3 |
| adamw | 0.001 | 1.66110 ± 0.00804 | 5.265 | 8917 | 10917 | 54.0 | 57.1 | 0 | 0 | 3/3 |
| dash_ndb | 0.001 | 1.66194 ± 0.01567 | 5.270 | 10417 | 12417 | 201.5 | 204.9 | 0 | 0 | 3/3 |
| dash_cn | 0.001 | 1.66239 ± 0.00312 | 5.272 | 13333 | 14917 | 242.7 | 246.9 | 0 | 0 | 3/3 |
| magma_muon | 0.03 | 1.66385 ± 0.01320 | 5.280 | 9000 | 11000 | 66.8 | 69.8 | 0 | 0 | 3/3 |
| adafisher | 0.001 | 1.66390 ± 0.00973 | 5.280 | 16167 | 17667 | 91.7 | 96.0 | 1 | 0 | 3/3 |
| magma_adam | 0.0009 | 1.66485 ± 0.01304 | 5.285 | 14000 | 16000 | 76.0 | 79.9 | 0 | 0 | 3/3 |
| adagrad | 0.03 | 1.66706 ± 0.00815 | 5.297 | 13333 | 15333 | 67.1 | 73.7 | 0 | 0 | 3/3 |
| muon | 0.03 | 1.66738 ± 0.01484 | 5.298 | 8333 | 10333 | 63.0 | 66.1 | 0 | 0 | 3/3 |
| muon_guarded | 0.3 | 1.67311 ± 0.01277 | 5.329 | 10417 | 12417 | 77.1 | 80.5 | 0 | 0 | 3/3 |
| sgd | 0.3 | 1.67311 ± 0.01277 | 5.329 | 10417 | 12417 | 53.6 | 56.9 | 0 | 0 | 3/3 |
| dash_ndb_guarded | 0.3 | 1.67311 ± 0.01277 | 5.329 | 10417 | 12417 | 205.1 | 208.8 | 0 | 0 | 3/3 |
| magma_adamw | 0.003 | 1.67749 ± 0.01368 | 5.352 | 8667 | 10667 | 55.9 | 58.9 | 0 | 0 | 3/3 |
| magma_sgd | 0.3 | 1.68991 ± 0.00887 | 5.419 | 14417 | 16083 | 73.3 | 77.2 | 0 | 0 | 3/3 |
| adamx | 0.003 | 1.70839 ± 0.01664 | 5.520 | 19750 | 20000 | 90.9 | 95.8 | 3 | 0 | 3/3 |
| amsgrad_geometric | 0.003 | 1.74475 ± 0.01645 | 5.724 | 19750 | 20000 | 93.6 | 98.5 | 3 | 0 | 3/3 |
| amsgrad_inverse | 0.003 | 1.81755 ± 0.00905 | 6.157 | 19917 | 20000 | 88.6 | 93.3 | 3 | 0 | 3/3 |

## Full-step compiler preflight

On the actual model/batch, each compiled case retained one FX graph, zero graph breaks and unchanged CUDA Graph node counts over 200 measured updates after 20 warmups. DASH EVD recorded two CUDA Graph nodes; the other methods recorded one. The only remaining compiler warning was `Not enough SMs to use max_autotune_gemm mode`. Initial mutable-input capture and Tensor-set problems were fixed before these measurements.

| Attention | Method | Eager ms/update | Full compile ms/update | Speedup | Cold compile s |
| --- | --- | ---: | ---: | ---: | ---: |
| softmax | adamw | 8.201 | 4.171 | 1.97x | 15.20 |
| softmax | magma_muon | 15.132 | 5.041 | 3.00x | 20.66 |
| softmax | dash_ndb_guarded | 39.843 | 14.600 | 2.73x | 27.08 |
| softmax | dash_evd | 26.590 | 15.251 | 1.74x | 13.13 |
| softmax | dash_chebyshev | 68.527 | 25.450 | 2.69x | 67.97 |
| softmax | adafisher | 12.264 | 4.434 | 2.77x | 16.04 |
| sparsemax | adamw | 9.885 | 4.291 | 2.30x | 13.44 |
| sparsemax | magma_muon | 17.377 | 5.161 | 3.37x | 16.21 |
| sparsemax | dash_ndb_guarded | 41.619 | 14.854 | 2.80x | 27.98 |
| sparsemax | dash_evd | 28.360 | 15.621 | 1.82x | 14.65 |
| sparsemax | dash_chebyshev | 69.534 | 25.567 | 2.72x | 70.68 |
| sparsemax | adafisher | 13.742 | 4.632 | 2.97x | 19.58 |

## Scope and artifacts

This ranking contains only this stopping protocol's runs, including all measured methods regardless of proof status. It is not combined with the earlier 1000-update last-iterate ranking. Those frozen sources/results remain intact, and their fingerprints are in metadata. Muon retains its recorded auxiliary Adam recipe; Magma retains Algorithm 1 with dense moments, p=.5, tau=2 and no 1/p. Convergence theorem assumptions are not certified by this experiment.

`runs.jsonl`: completed runs, all validation curves, realized budgets, reasons, hashes, memory and timing. `summary.json`/`summary.csv`: rankings. `metadata.json`: stopping settings, environment, data/source hashes and tests. `validation.json`: replayed decisions and CUDA checkpoint re-evaluation. Best model checkpoints and live curves are under ignored `experiments/runs/full_compile_benchmark/`. Resume restarts an unfinished run and skips completed identifiers under the same fingerprint.

```bash
.venv/bin/python -u -m experiments.full_compile_benchmark --all
.venv/bin/python -m experiments.full_compile_benchmark --all --validate-only
```
