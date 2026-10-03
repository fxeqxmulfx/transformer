# Magma on GPTMini and Tiny Shakespeare

## Predeclared protocol

This follow-up tests arXiv:2602.15322v1, Algorithm 1 and Sections 3–4, on
the existing small GPTMini experiment. Read the original local source at
`papers/arXiv-2602.15322v1/google_main.tex`. This is a new dataset/model
experiment, not a reproduction of the paper's C4/Llama training schedule.

- [x] Write and pass analytic tests before any training, including CUDA smoke.
- [x] Verify the frozen old sources and results without editing them.
- [x] Calibrate all new methods on the RTX 3050 Laptop GPU.
- [x] Screen each declared rate for 250 updates on seed 0, for both attentions.
- [x] Select rates using final validation CE only; never use test loss.
- [x] Train selected rates for 1000 updates on seeds 0, 1, 2.
- [x] Verify paired initialization and batch hashes against the old experiment.
- [x] Reload checkpoints, evaluate test loss, and audit masks/moments.
- [x] Save all raw runs, source/data hashes, tests, and combined rankings.

Use the unchanged `experiments/gpt_mini.py`, frozen attention/data/training
engine, Tiny Shakespeare 90/5/5 character split, 2 layers, 4 heads, width
128, FFN 512, context 64, batch 32, float32, deterministic algorithms,
TF32 disabled. Initialization, full held-out evaluation, and minibatch
streams are exactly those of `optimizer_benchmark/results/rtx3050`.
The old 18 methods remain in the primary ranking, regardless of proof status.
Their raw measurements and measured Python sources must remain unchanged.
New code and output live outside the frozen `optimizer_benchmark` package.

## Magma recipe and declared learning rates

Each attention/FFN weight tensor is one independent Bernoulli(0.5) block:
eight matrices, 393,216 parameters. The tied embedding/head matrix and
two temperature vectors (8,328 parameters) keep their dense base updates.
There are no trainable RMSNorm scales. Fused QKV is one parameter block.

First compute every base direction and update all its momentum/variance
states. Then compute the cosine of the **new** dense first moment with
the current gradient; sigmoid(cosine / 2); scalar EMA with coefficient
0.9; multiply the complete base displacement by the new scale and mask.
Algorithm 1 has **no 1/p correction**. AdamW decay is part of this complete
displacement, and is therefore masked/damped on eligible matrices.

Initialize the scalar EMA to 0.5 and define zero-vector cosine as zero.
The paper does not specify these two conventions; both match the local
Lean extension. Mask RNG has its own CPU generator seeded with 20000+seed,
independent of model initialization and training-window RNG. Record actual
per-block survival and damping statistics. Preserve RNG and dense state in
optimizer checkpoints.

| New method | Base and role | Declared rates |
| --- | --- | --- |
| rmsprop | Raw second-moment RMSProp control | 0.0001, 0.0003, 0.001 |
| magma_rmsprop | Paper's main Magma pairing | 0.0003, 0.001, 0.003 |
| magma_adam | Frozen raw-moment Adam + Magma | 0.0009, 0.003, 0.009 |
| magma_adamw | Frozen bias-corrected AdamW + Magma | 0.0009, 0.003, 0.009 |
| magma_muon | Frozen Muon/auxiliary-Adam hybrid + Magma | 0.01, 0.03, 0.1 |
| magma_sgd | Frozen raw-gradient SGD + Magma | 0.1, 0.3, 1.0 |

All new methods receive exactly three candidates, 750 screening updates
per attention, and the same 3000 final updates. Adaptive Magma grids extend
above the corresponding dense grids because the hidden-matrix expected
displacement is reduced to roughly one quarter. Rates on unmasked weights
remain the base rates; this matters when interpreting the results. Grids
are fixed before any Magma training and are not expanded after test scores.

RMSProp uses v=0.999*v+0.001*g², direction g/(sqrt(v)+1e-8), no debiasing,
no momentum in its direction, and zero decay. It maintains an additional
0.9 first-moment EMA for Magma scoring; this adds memory, explicitly.
SGD likewise maintains a scoring EMA without changing its raw-gradient
base direction. Adam/AdamW reuse beta1=0.9 momenta. Muon reuses its
beta=0.95 raw momentum, whose constant positive scaling cancels in cosine.
Muon uses the previously measured five-step NS, shape scaling, 0.05
auxiliary LR ratio and zero decay. Auxiliary Adam is still an empirical
control; this wrapper does not prove its general convergence.

The corrected Lean convergence result is for normalized masked **SGD**
under explicit smoothness, sampling, and step assumptions. Algorithm 1
at p=0.5 differs by a factor of two in the learning rate; this benchmark
does not certify those assumptions or convergence of adaptive bases.

## Execution and artifacts

Run tests first and train only on CUDA:

```bash
OPTIMIZER_BENCH_CUDA=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m unittest discover -s experiments/magma_benchmark/tests -v
.venv/bin/python -u -m experiments.magma_benchmark --calibrate
.venv/bin/python -u -m experiments.magma_benchmark
```

Output: `experiments/magma_benchmark/results/rtx3050/`.
Seed-0 checkpoints: ignored `experiments/runs/magma_benchmark/rtx3050/`.
Resume only when data/source/protocol fingerprints agree; completed runs
are retained and skipped. Comparison report includes all 24 measured
methods, per-seed paired deltas, sample SD, timing and memory. Measurements
from the old session retain their original timing and provenance.

## Execution log

- Tests were written first; the initial run failed because the implementation
  module did not yet exist. Nine analytic CPU tests then passed after implementation.
- 2026-10-01 08:08 UTC: all 60 combined CPU/CUDA tests passed, zero skips.
  All 12 calibration runs passed; peak allocated memory was 147 MiB.
- Frozen baseline verification succeeded: 108 final measurements and all
  original source fingerprints remained unchanged.
- 2026-10-01 08:09–08:18 UTC: all 36 screening and 36 final runs completed,
  with zero failures, totaling 45,000 updates. All 60 tests passed again
  before the full comparison. Peak allocated memory was 146.94 MiB.
- Validation selected RMSProp rates 0.0001 (softmax), 0.0003 (Sparsemax);
  Magma+RMSProp 0.0003; Magma+Adam 0.0009; Magma+AdamW 0.003;
  Magma+Muon 0.03; Magma+SGD 0.3 for both attention types.
- All 12 seed-0 checkpoints reproduced their recorded test CE on CUDA.
  Init/batch hashes match the old experiment; mask streams match across
  Magma variants and attentions. The old raw log and sources remain intact.
- Magma+Muon has the lowest observed mean among all 24 methods: softmax
  1.75293 ± 0.00365 (PPL 5.772), Sparsemax 1.78222 ± 0.00687 (PPL 5.943).
  The previous AdamW means were 1.76013 and 1.78993. Muon-base paired
  CE deltas are -0.01510 (3/3 seeds improved) and -0.01049 (2/3).
- Other Magma pairings did not improve their base's three-seed mean at this
  budget. In particular, the paper's RMSProp+Magma advantage did not
  reproduce here. These are short GPTMini/Tiny Shakespeare measurements,
  not a general optimizer ordering or a convergence proof.
- Full results: [REPORT.md](results/rtx3050/REPORT.md), with all raw values,
  paired deltas, checkpoints, tests, and combined 24-method summaries.
