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

All nine CUDA runs completed 10,000 updates. Every final train and held-out
accuracy is 100%. The table reports epochs (ten updates per epoch). The
last column is the first epoch after which every recorded observation stays
at or above 99% through epoch 1,000; it is a finite-budget observation.

| Branch | First held-out 99%, seeds 11 / 17 / 23 | Worst held-out accuracy after first 99% | At least 99% thereafter through the budget |
| --- | --- | --- | --- |
| ReLU | 29 / 31 / 30 | 99.08% / 99.38% / 99.31% | 29 / 31 / 30 |
| SiLU | 28 / 26 / 27 | 99.18% / 99.27% / 99.31% | 28 / 26 / 27 |
| SwiGLU | 14 / 15 / 16 | 94.05% / 95.25% / 86.10% | 967 / 926 / 909 |

SwiGLU reaches the first crossing about twice as quickly but has 24, 14 and
12 later observations below 99%. At its worst held-out observations,
training accuracy also falls to 93.84%, 95.88% and 86.13%. These are losses
of fit on both splits. ReLU and SiLU have no later observations below 99%.
Thus a first-crossing comparison alone would conceal the gated model's
instability at the ReLU baseline's learning rate. The
[step-size controls](../rotate90_swiglu_stability/README.md) find no such
relapses at 0.003 or 0.001 through nearly equal nominal optimizer times.
An analytic activation alone does not supply that stability guarantee.

The learned embeddings also develop a specific geometry. They begin with
participation rank 53.78--54.88 and J overlap 0.631--0.646.

| Branch | Final participation rank | Final J overlap | Character pairs carrying 90% of centered E energy |
| --- | --- | --- | --- |
| ReLU | 19.43--21.20 | 0.091--0.167 | 9--10 |
| SiLU | 18.14--20.84 | 0.072--0.137 | 9--10 |
| SwiGLU | 8.41--13.64 | 0.027--0.032 | 5--7 |

Predetermined checkpoint ablations retain 0, 1, 2, 4, 8, 16 or all 48
conjugate character pairs, chosen only by E's energy. They keep its row
mean and zero row, with every branch matrix fixed, and replace all three
tied uses of E together. Keeping eight pairs preserves 100% held-out
accuracy in every run; training accuracy is 100% except for one missed row
in `relu-s23` (99.9785%). Keeping all 48 reconstructs every FP32 parameter
exactly and reproduces both recorded losses and accuracies. Four pairs
already preserve at least 99.44% held-out accuracy in eight runs, but only
78.67% in `swiglu-s23`. Concentration in one dominant character is therefore
an inadequate description of the trained networks.

These results support a character representation with separated operand
subspaces. They do not yet explain how typical initialization finds it,
prove that the character projection preserves the real-valued function, or
resolve the scalar MLP's gradient-access question. The comparison remains
at one fixed prime with known operand values; it does not test new numbers.

Portable evidence is in [results.json](results.json), the full 9,009-row
performance and geometry curves in [curves.csv](curves.csv), and the exact
lab file hashes in [engine_code.json](engine_code.json). Raw checkpoints,
per-update gradients and full character spectra remain under `runs/`; their
hashes are exported. The runs began at `ae7e920d` with the lab changes later
committed as `c59d4426`; the recorded engine hashes match that committed
implementation. Analysis verifies the experiment, engine, complete budget,
every checkpoint's original metrics and the full-reconstruction control.
Runtime was recorded during concurrent CPU regression testing; the
comparisons here use update counts.

To regenerate the portable artifacts from the retained runs, run this from
`python/` with the recorded lab implementation and PyTorch runtime:

```sh
uv run --locked python ../experiments/rotate90_analytic/analyze.py
```

## Constructive product witness

The separate [expressivity script](expressivity.py) constructs a checkpoint
of the unchanged width-128, H=43 SwiGLU model using a faithful multiplicative
character, a constant feature and an explicit zero-residue feature. Nine
products, realized by 18 individual channels in opposed pairs, cancel
the unwanted skip coordinates and form the quotient character. All 43
gates remain in the actual model, with 25 set to zero. Both RMSNorms, the
residual and all three tied uses of E are retained.

All 9,312 equations classify correctly in CPU float64, CPU float32 and
CUDA float32, including the 96 zero-numerator rows. Every parameter has
absolute value at most one. The minimum correct-class margin is about
0.02409; CE is about 2.42165, so correct classification is not zero loss.
Float64's maximum raw state and logit errors against the intended E[q]
are 1.89e-15 and 2.31e-14; the FP32 checks remain below 6e-7 and 5e-6.
[division_witness.json](division_witness.json) retains every check and hash.

`Neural.Products` proves the exact SiLU difference, paired-gate product,
and their compilation into the actual full residual state and tied logits.
The complete character checkpoint is numerically checked here; a Lean proof
of that entire finite-table checkpoint is still a separate obligation.
This certificate establishes existence of a bounded low-width solution,
not attraction from the random initialization, and is not used to initialize
any learning run. Its single character differs from the multi-character
geometry of the trained networks measured above. The
[statement review](../../review/rotate90_products.md) gives the formulas.

```sh
uv run --locked python ../experiments/rotate90_analytic/expressivity.py
```
