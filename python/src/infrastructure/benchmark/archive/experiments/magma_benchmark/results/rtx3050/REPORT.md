# Magma on GPTMini and Tiny Shakespeare

GPU: **NVIDIA GeForce RTX 3050 Laptop GPU**, 3.68 GiB CUDA-visible memory; Torch 2.14.0+cu130, CUDA 13.0, float32, TF32 disabled.

All 60 CPU/CUDA tests passed before the experiment, with zero failures/skips.

## Observed ranking

The primary ranking contains every measured method: 18 frozen baseline methods and six new methods. It is ordered by the mean final test cross-entropy of three seeds. ± is sample SD, not a confidence interval; lower CE/perplexity is better.

**softmax: magma_muon** has the lowest observed mean: 1.75293 ± 0.00365, character perplexity 5.772.

### softmax

| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| magma_muon | 0.03 | 1.75293 ± 0.00365 | 5.772 | 14.7 | 7.136 | 142 | 3/3 |
| adamw | 0.001 | 1.76013 ± 0.01341 | 5.813 | 8.2 | 1.007 | 145 | 3/3 |
| adam | 0.0003 | 1.76436 ± 0.00949 | 5.838 | 7.9 | 0.617 | 145 | 3/3 |
| amsgrad | 0.0003 | 1.76445 ± 0.00963 | 5.838 | 8.2 | 0.752 | 145 | 3/3 |
| muon | 0.03 | 1.76803 ± 0.00932 | 5.859 | 12.6 | 5.249 | 142 | 3/3 |
| magma_adamw | 0.003 | 1.77172 ± 0.00276 | 5.881 | 10.1 | 2.745 | 145 | 3/3 |
| magma_adam | 0.0009 | 1.77788 ± 0.01476 | 5.917 | 9.7 | 2.365 | 145 | 3/3 |
| adamnc | 0.01 | 1.79868 ± 0.00712 | 6.042 | 7.9 | 0.626 | 145 | 3/3 |
| dash_cn | 0.001 | 1.80401 ± 0.01121 | 6.074 | 35.6 | 28.118 | 146 | 3/3 |
| dash_evd | 0.001 | 1.80403 ± 0.01119 | 6.074 | 26.5 | 18.982 | 233 | 3/3 |
| dash_chebyshev | 0.001 | 1.80409 ± 0.01114 | 6.074 | 67.0 | 59.291 | 146 | 3/3 |
| dash_ndb | 0.001 | 1.80746 ± 0.00380 | 6.095 | 37.8 | 30.263 | 146 | 3/3 |
| rmsprop | 0.0001 | 1.83207 ± 0.00484 | 6.247 | 7.9 | 0.513 | 143 | 3/3 |
| muon_guarded | 0.1 | 1.85975 ± 0.00619 | 6.422 | 13.5 | 6.172 | 142 | 3/3 |
| sgd | 0.1 | 1.85975 ± 0.00619 | 6.422 | 7.7 | 0.199 | 145 | 3/3 |
| dash_ndb_guarded | 0.1 | 1.85975 ± 0.00619 | 6.422 | 38.3 | 30.823 | 146 | 3/3 |
| magma_rmsprop | 0.0003 | 1.86285 ± 0.02011 | 6.442 | 9.6 | 2.243 | 143 | 3/3 |
| adafisher | 0.001 | 1.88625 ± 0.00620 | 6.595 | 11.9 | 3.379 | 142 | 3/3 |
| adafisherw | 0.001 | 1.88684 ± 0.00859 | 6.598 | 11.9 | 3.576 | 142 | 3/3 |
| adagrad | 0.03 | 1.91586 ± 0.02837 | 6.793 | 8.3 | 0.799 | 145 | 3/3 |
| magma_sgd | 0.3 | 1.93264 ± 0.00386 | 6.908 | 9.3 | 1.936 | 146 | 3/3 |
| adamx | 0.003 | 1.98792 ± 0.01824 | 7.300 | 8.2 | 0.872 | 145 | 3/3 |
| amsgrad_geometric | 0.003 | 1.99193 ± 0.01929 | 7.330 | 8.0 | 0.739 | 145 | 3/3 |
| amsgrad_inverse | 0.003 | 2.18394 ± 0.07320 | 8.881 | 7.9 | 0.643 | 145 | 3/3 |

**sparsemax: magma_muon** has the lowest observed mean: 1.78222 ± 0.00687, character perplexity 5.943.

### sparsemax

| Method | LR | Test CE ± SD | Char. PPL | Train s | Optimizer ms/step | Peak MiB | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| magma_muon | 0.03 | 1.78222 ± 0.00687 | 5.943 | 15.9 | 6.975 | 142 | 3/3 |
| adamw | 0.001 | 1.78993 ± 0.00632 | 5.989 | 9.6 | 0.899 | 145 | 3/3 |
| adam | 0.0003 | 1.79137 ± 0.01484 | 5.998 | 9.3 | 0.523 | 145 | 3/3 |
| magma_adamw | 0.003 | 1.79177 ± 0.01488 | 6.000 | 11.6 | 2.723 | 145 | 3/3 |
| muon | 0.03 | 1.79271 ± 0.00529 | 6.006 | 14.0 | 5.259 | 142 | 3/3 |
| magma_adam | 0.0009 | 1.79979 ± 0.00845 | 6.048 | 11.3 | 2.336 | 145 | 3/3 |
| amsgrad | 0.0003 | 1.80118 ± 0.00369 | 6.057 | 9.4 | 0.641 | 145 | 3/3 |
| adamnc | 0.01 | 1.81425 ± 0.00081 | 6.136 | 9.2 | 0.476 | 145 | 3/3 |
| rmsprop | 0.0003 | 1.82302 ± 0.01247 | 6.191 | 9.3 | 0.428 | 144 | 3/3 |
| dash_evd | 0.001 | 1.83175 ± 0.00851 | 6.245 | 27.2 | 18.418 | 233 | 3/3 |
| dash_chebyshev | 0.001 | 1.83345 ± 0.00489 | 6.255 | 67.1 | 58.109 | 147 | 3/3 |
| dash_cn | 0.001 | 1.83430 ± 0.00546 | 6.261 | 36.9 | 27.919 | 147 | 3/3 |
| dash_ndb | 0.001 | 1.83653 ± 0.01149 | 6.275 | 39.4 | 30.355 | 147 | 3/3 |
| magma_rmsprop | 0.0003 | 1.87132 ± 0.01274 | 6.497 | 11.2 | 2.282 | 144 | 3/3 |
| adagrad | 0.03 | 1.87897 ± 0.01405 | 6.547 | 9.2 | 0.511 | 145 | 3/3 |
| dash_ndb_guarded | 0.3 | 1.90579 ± 0.08004 | 6.725 | 39.4 | 30.513 | 147 | 3/3 |
| sgd | 0.3 | 1.90579 ± 0.08004 | 6.725 | 9.0 | 0.143 | 145 | 3/3 |
| muon_guarded | 0.3 | 1.90579 ± 0.08004 | 6.725 | 14.9 | 6.080 | 142 | 3/3 |
| magma_sgd | 0.3 | 1.93171 ± 0.00915 | 6.901 | 11.2 | 2.034 | 147 | 3/3 |
| adafisher | 0.001 | 1.93403 ± 0.02556 | 6.917 | 13.0 | 3.278 | 142 | 3/3 |
| adafisherw | 0.001 | 1.93792 ± 0.02202 | 6.944 | 13.1 | 3.441 | 142 | 3/3 |
| amsgrad_geometric | 0.003 | 1.97968 ± 0.01289 | 7.240 | 9.2 | 0.504 | 145 | 3/3 |
| adamx | 0.003 | 1.98208 ± 0.00226 | 7.258 | 9.6 | 0.771 | 145 | 3/3 |
| amsgrad_inverse | 0.003 | 2.10723 ± 0.01781 | 8.225 | 9.4 | 0.661 | 145 | 3/3 |

## Magma versus its dense base

Each base and wrapper selects its rate using its declared validation-only search. The deltas below pair initialization and minibatches, but allow different selected rates. A negative delta favors Magma. Three seeds do not establish a general ordering.

| Attention | Wrapper | Mean CE delta | Paired delta SD | Improved seeds | Seed deltas |
| --- | --- | ---: | ---: | ---: | --- |
| softmax | magma_rmsprop | +0.03078 | 0.01564 | 0/3 | +0.01557, +0.04683, +0.02993 |
| softmax | magma_adam | +0.01352 | 0.00527 | 0/3 | +0.01860, +0.01387, +0.00808 |
| softmax | magma_adamw | +0.01159 | 0.01611 | 1/3 | +0.00675, -0.00154, +0.02957 |
| softmax | magma_muon | -0.01510 | 0.00930 | 3/3 | -0.02385, -0.01610, -0.00535 |
| softmax | magma_sgd | +0.07289 | 0.00526 | 0/3 | +0.06728, +0.07773, +0.07367 |
| sparsemax | magma_rmsprop | +0.04829 | 0.00529 | 0/3 | +0.04555, +0.04494, +0.05439 |
| sparsemax | magma_adam | +0.00842 | 0.02086 | 1/3 | +0.02275, -0.01551, +0.01801 |
| sparsemax | magma_adamw | +0.00184 | 0.00904 | 1/3 | +0.01099, +0.00162, -0.00710 |
| sparsemax | magma_muon | -0.01049 | 0.01191 | 2/3 | +0.00175, -0.01119, -0.02204 |
| sparsemax | magma_sgd | +0.02592 | 0.08888 | 1/3 | +0.06808, +0.08587, -0.07619 |

## Recipe, scope, and reproducibility

- Local source: `papers/arXiv-2602.15322v1/google_main.tex`, Algorithm 1 and Sections 3–4. Independent Bernoulli(0.5) masks on eight attention/FFN matrices; tau=2; score EMA=.9; first moments and variances updated densely before masking. The fused QKV tensor is one block. The 393,216 masked parameters exclude the 8,328 embedding/head/temperature parameters.
- Apply s*m to the entire base displacement, including AdamW decay, without 1/p. Initial scale=.5 and zero-vector cosine=0 are explicit extensions where the paper is silent. SGD/RMSProp retain an extra dense scoring EMA. Adam/AdamW/Muon reuse their base first moment. Independent CPU mask RNG uses 20000+seed; all Magma methods share mask streams.
- Unmodified GPTMini: 2 layers, 4 heads, width128, FFN512, context64, batch32, 401,544 unique parameters, 65 characters, contiguous 90/5/5 split. All original normalization, XSA, RoPE and tying remain. Repeat with softmax and causal Sparsemax.
- Each new method receives three predeclared rates, 250 screening updates per rate on seed0, then 1000 final updates on seeds0,1,2. No warmup, LR schedule, clipping, AMP, dropout or compilation. Full fixed-window held-out evaluation; no test-loss tuning. Grids and conventions are in `../../PLAN.md`.
- RMSProp: raw v EMA=.999, epsilon=1e-8, no momentum in the direction, no debiasing/decay. Adam is the old raw-moment rule; AdamW is bias corrected, decay=.01. Muon preserves the old five-step Newton–Schulz and auxiliary Adam on tied embeddings/temperatures, with LR ratio .05 and zero decay. This remains the documented experimental hybrid, rather than the paper's shared-LR recipe.
- The corrected Lean stationarity theorem is for normalized masked SGD under explicit smoothness, sampling and step-size hypotheses. It does not prove convergence of these adaptive wrappers or certify GPTMini training assumptions. Sparsemax attention does not make joint training convex.
- This short Tiny Shakespeare experiment differs from C4/Llama and the paper's warmup/cosine schedule. Conclusions apply to these measured budgets and grids. New and old timing come from separate sessions on the same GPU; old timings retain their original values.

## Artifacts

`runs.jsonl` stores every new screening/final run, validation curves, masks/damping, timings, memory and paired hashes. `selected_rates.json` records validation-selected rates. `summary.*` contains the new methods; `combined/summary.*` contains all methods. `paired_deltas.json`, `validation.json`, calibration and test logs preserve the checks. Seed0 checkpoints include model, optimizer states and mask RNG under the ignored `experiments/runs/magma_benchmark/rtx3050/`.

Frozen baseline raw SHA256: `01b42cb9e0e42c2ac04173d9c2c723eb75c2bf8cd4e7c9a7654ab19fb74f6f47`. New protocol SHA256: `c2aab044822615d425c60681dcb33ebe24e6a82192dd578f46a84263bc090992`. Both sets of measured sources are fingerprinted in `metadata.json`; resume refuses mismatched fingerprints.

```bash
.venv/bin/python -u -m experiments.magma_benchmark --calibrate
.venv/bin/python -u -m experiments.magma_benchmark
.venv/bin/python -m experiments.magma_benchmark --validate-only
```
