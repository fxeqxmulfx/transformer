# FP4 stochastic rounding with upward E4M3 scales

Source: arXiv:2601.22813v2, §3.1, §3.3, Algorithm 1, and Table 1. The
corrected scale policy is formalized in
`src/Transformer/Quartet/Section3_SRCeilScale.lean`.

## Scale policy and guarantee

For each rotated chunk `y` of 128 values, choose the FP32 tensor scale
`T = max|y|/(6·448)` when `y ≠ 0` (and `1/448` otherwise). For each group of
16 values, choose the smallest E4M3 value at least `max_g|y|/(6T)`; a zero
group uses the representable E4M3 value `1`. Round **every FP4 element**
stochastically with its own coin. The Lean theorem
`integral_rhtInv_qSRCeilScale` proves that, for every input and every fixed
RHT seed, the expected inverse-rotated vector equals the input. The tensor
scale is an exact real in the Lean model, as in the paper formalization.

## Error measurement

The reference evaluator computes the exact rounding-coin moments from the two
adjacent grid values. If `z = y/(gT)` lies between FP4 values `a ≤ z ≤ b`,
then the dequantized SR value has mean `y` and squared error
`(y − gTa)(gTb − y)`. For MS-EDEN, the evaluator uses Algorithm 1's RTN FP4,
per-group EDEN factor, and SR FP8 correction, then adds squared conditional
bias and variance. It samples only the Gaussian input vectors. Orthogonality
and rotation invariance of a standard Gaussian make the rotated-input MSE
equal to the inverse-rotated MSE.

The run used 100,000 independent `N(0,1)` vectors of dimension 128, seed
`20260929`. Entries are per-coordinate MSE; `SE` is the standard error across
sampled vectors. Table 1's values are included as a cross-check.

| Method | MSE | SE | Table 1 MSE |
| --- | ---: | ---: | ---: |
| Paper `Q_SR`, RTN E4M3 groups | 0.023611 | 0.000010 | 0.0235 |
| Corrected SR, upward E4M3 groups | 0.021283 | 0.000010 | — |
| SR, constant E4M3 group scale `1` | 0.024048 | 0.000011 | — |
| MS-EDEN, paper clipping factor | 0.009295 | 0.000005 | 0.0094 |

The corrected SR policy reduces Gaussian MSE by about 9.9% relative to the
paper's `Q_SR` baseline, while its MSE is 2.29 times that of MS-EDEN. The
MS-EDEN row measures error, not the validity of its exact unbiasedness claim:
the same reference evaluator reproduces the formal counterexample's mean
first coordinate `101/102` for `e₀ + e₁/10` at dimension 128.

## Speed and cost

Each 128-entry chunk uses 128 independent FP4 rounding coins with the
corrected SR policy. Algorithm 1 uses 8 FP8 group-scale rounding coins. That
is a 16-fold difference in random draws, before implementation details.

The NumPy reference measured median **host CPU quantizer-stage** times on
4,096 chunks (7 repetitions): 40.94 ms for corrected SR, 40.71 ms for the
paper's SR, and 44.64 ms for MS-EDEN. These timings include NumPy allocation
and grid lookups; they exclude RHT, GEMM, transfers, and GPU kernels. They do
not predict Blackwell throughput. The corrected policy has the same per-entry
SR work as the paper's SR baseline and changes only FP8 scale selection; this
suggests similar GPU quantizer cost, but remains an inference. No working
NVIDIA driver, PyTorch, or
Triton is available in this workspace, so the hardware kernel-speed claim
cannot be measured here. A GPU benchmark must use the actual packed NVFP4
kernels and matched GEMM shapes before a training-throughput claim is made.

## Reproduction

```sh
python3 experiments/quartet_sr_evaluation/quartet_sr_evaluation.py \
  --samples 100000 --batch 4096 --repeats 7 \
  --json experiments/quartet_sr_evaluation/quartet_sr_evaluation.json
```

The script checks the FP4 SR moment at `0.75`, zero and underflowing groups,
FP8 representability and non-clipping of the corrected scales, and the
MS-EDEN `101/102` witness before printing the results. The output is stored
in [`quartet_sr_evaluation.json`](quartet_sr_evaluation.json). The script
needs only NumPy and predates the [lab](../../python/README.md).
