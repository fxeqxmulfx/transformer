# Analytic activation controls for the Rotate90 grokking lab

Does replacing the published model's ReLU with SiLU or a near-budget SwiGLU
branch improve generalization on the same modular division task? This is
M9.3 of [the Landscape research plan](../../landscape_grokking_plan.md).

The source is `grokking_abmod/web/own-channel/python/rotate90.py` and
`README.md` at `f49969393a7482490d45483d8a4fd3046b1adede`, the W128 Fast
preset of the [published lab](https://fxeqxmulfx.github.io/grokking_abmod/).
The shared E, pairwise quarter-turn, residual, pre-branch and final RMSNorm
with epsilon 1e-5, bias-free matrices and 97-way cross-entropy are retained.

| Run prefix | Residual branch on normalized r | Intermediate width | Parameters | Change from source |
| --- | --- | ---: | ---: | --- |
| relu | ReLU(r A) | 128 | 28,800 | Baseline |
| silu | SiLU(r A) | 128 | 28,800 | Scalar activation only |
| swiglu | (SiLU(r G) * (r U)) D | 43 | 28,928 | Three projections and multiplicative gate |

Each has seeds 11, 17 and 23. Within a seed, initialization E, the 4,656-row
train split, 4,656-row held-out split and every minibatch order agree.
The SiLU control also starts with exactly the baseline A. E starts with
variance 1/128; A and G with 2/128, U with 1/128 and D with 1/43. Matrices
use independent sequential Box-Muller draws from the source's SplitMix64
stream. SwiGLU's additional projections change both its parameterization
and initialization; this is recorded rather than attributed to analyticity.

The baseline's seed-11 checkpoint, both split fingerprints and 21 minibatches
across three epochs match independently recorded source fixtures. Tests
also check complete tied gradients by finite differences and against three
independent copies of E for its three roles. An unmodified source export is
vendored in `third_party/rotate90/reference.py`. On the current CPU runtime,
20 complete updates through the lab match that export exactly, including
warmup, short epoch tails, every evaluated loss and accuracy, and every final
parameter. This reference computes its own initialization, split and updates.

Training uses AdamW, learning rate 0.01, decay 1, betas (0.9, 0.98), epsilon
1e-8 and ten inclusive warmup updates. Nine batches of 512 and one of 48
make an epoch. The 10,000-update budget is 1,000 complete epochs; both
splits are evaluated every epoch, and checkpoints save every 100 epochs.
Every update records the gradient norm; observations record parameter,
embedding and zero-embedding norms. Each observation also orders the nonzero
embedding rows by powers of the smallest generator (5 for modulus 97) and
records the centered geometry:

- Participation rank: `tr(X X.T)^2 / ||X X.T||_F^2`.
- Operand overlap: `||X (J X).T||_F / ||X X.T||_F`. Zero means that the two
  centered operand subspaces are orthogonal; one includes a J-invariant plane.
- Common-row energy and every conjugate Fourier pair's share of nonconstant
  energy, with the dominant character's frequency and order.

These observers use detached float64 values and do not change updates. Tests
distinguish a position-separated faithful character from a J-invariant one
and handle a constant embedding without inventing ratios. Crossings in the
result use updates; divide by ten to compare with browser epochs.

The dense branch uses 16,384 matrix multiply-accumulates per example and the
SwiGLU projections use 16,512, before their shared readout. These counts do
not price activation functions, normalization or the optimizer; runtime is
measured rather than described as an equal-FLOP comparison.

Deliberate backend differences: this uses the lab's locked PyTorch
2.14.1+cu126 and its eager CUDA AdamW implementation, rather than the
source export's PyTorch 2.10 CPU loop or the native/browser FP32 kernels.
All variants share this backend. Training continues for the whole budget
after the first 99% crossing; the browser pauses there. First crossings
alone do not certify the browser's plateau-and-rise grokking criterion.

The Lean models in `Transformer.Landscape.Neural.SiLU` and `.SwiGLU`
(commit `00d36dd8`) prove analytic loss functions in these actual shared
parameters and the local Lojasiewicz inequality. They do not prove that
the models generalize or that AdamW follows the M8 gradient-descent orbit.

```sh
./run.py check experiments/rotate90_analytic
./run.py show experiments/rotate90_analytic swiglu-s11
./run.py run experiments/rotate90_analytic
./run.py report experiments/rotate90_analytic
```

## Findings

The source port passes its independently recorded initial checkpoint,
train/test split and continuing sampler fixtures. All three branches pass
finite-difference checks for every parameter, and the tied E gradient equals
the sum of separate operand-a, operand-b and readout gradients. A complete
small training run also passes through the lab's sampler, optimizer,
evaluation and checkpoint interfaces. The complete 20-update source replay
and the independently specified geometry examples pass on the updated stack.

The nine CUDA runs are prepared; their training results will be added here.
Analyticity alone is not evidence of improved training.
