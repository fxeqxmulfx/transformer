# Matched modular-division MLP controls

Which part of the scalar MLP's difficulty comes from its input code, its
answer code, its loss or AdamW decay? These controls use the browser's exact
division equations, half-table split and continuing sampler, while changing
one of those choices at a time. They complement the
[Rotate90 controls](../rotate90_analytic/README.md); they do not isolate every
architectural difference between a plain MLP and Rotate90.

## Runs

Every model is a plain MLP with two biased SiLU hidden layers of width 195.
The output is an untied affine map. Scalar operands are normalized to
`2v/96 - 1`; one-hot operands are concatenated into 194 coordinates.

| Label prefix | Input coordinates | Output coordinates | Objective per equation | Parameters |
| --- | ---: | ---: | --- | ---: |
| `scalar-number-mse` | 2 | 1 | half squared error to `2q/96 - 1` | 39,001 |
| `onehot-number-mse` | 194 | 1 | half squared error to `2q/96 - 1` | 76,441 |
| `scalar-vector-mse` | 2 | 97 | half squared Euclidean error to `e_q` | 57,817 |
| `onehot-vector-mse` | 194 | 97 | half squared Euclidean error to `e_q` | 95,257 |
| `scalar-vector-ce` | 2 | 97 | full 97-class cross-entropy | 57,817 |
| `onehot-vector-ce` | 194 | 97 | full 97-class cross-entropy | 95,257 |

Each row has decay `wd0` or `wd1` and seeds `s11`, `s17`, `s23`: 36 runs.
The vector squared loss sums coordinates; it does not divide by 97. Both
squared objectives include a factor of one half, unlike the older
[whole-table regression](../mod97_scalar_mlp/README.md). Cross-entropy uses
raw logits and has no EOS term. Changing vector MSE to CE preserves the
entire model and its initial weights.

All runs use AdamW with learning rate 0.003, betas (0.9, 0.98), epsilon
1e-8, inclusive ten-update linear warmup, batch size 512 and 30,000
updates. Decay applies to every parameter, including biases. There are
4,656 training and 4,656 held-out equations; one continuing source epoch
has nine full batches and a final batch of 48. All training budgets are
3,000 such epochs. Evaluation and gradient/update observations occur every
100 updates, or ten epochs, and checkpoints every 1,000 updates. The
coarser evaluation cadence must be retained when reporting first crossings.
The Eager CUDA backend is the locked lab backend.

`TorchDefault` initializes each network. For a fixed input code and seed,
the scalar and vector heads share every hidden parameter draw. Both vector
objectives also share their output parameter draws. Input widths change
PyTorch's fan-in initialization and the parameter count; these are fixed
hidden-width controls, not matched-capacity or matched-initial-kernel
controls. Observations include first-layer preactivation/activation RMS on
the fixed first 512 training equations to expose that conditioning change.
The Rotate90 comparisons additionally differ in tied weights, residual
connections and normalization, so an MLP/classifier comparison alone cannot
attribute the outcome to any one of those mechanisms.

## Decoding and expressivity

Scalar answers are decoded by rounding `(f + 1) * 48`, without clipping to
the vocabulary. Vector answers use argmax, with PyTorch's smallest-index
tie policy. Every run reports train and held-out accuracy, objective loss
and decoded numeric MSE in residue-value units. Scalar runs additionally
retain continuous numeric MSE before rounding. The losses have different
scales and must not be compared as if they were the same accuracy measure.

An independent multiplication oracle checks every data row. Tests check
the original rows, splits and 21 consecutive batches against `LabDivision`
for all three seeds, loss and decoding formulas on a small exact example,
every parameter's float64 finite-difference gradient, shared initialization,
and the actual engine/checkpoint path for all three objectives. A finite
two-layer SiLU scalar lookup construction at width 195 answers all 9,312
equations, with raw numeric MSE below 1e-8. That construction only certifies
expressivity in the test; training always starts from random parameters
and never receives held-out answers or constructed teacher weights.

## Status

The experiment is prepared; no training outcome is recorded yet. A failure
to learn within this finite budget will distinguish observed fitting from
generalization, and will not prove that gradient training can never reach
a good solution. Quantitative gradient-access and attraction claims remain
the next Landscape obligation.

```sh
./run.py check experiments/mod97_representation_controls
./run.py show experiments/mod97_representation_controls scalar-number-mse-wd0-s11
./run.py run experiments/mod97_representation_controls
./run.py report experiments/mod97_representation_controls
```

Protocol source: `/home/misha/rust_projects/grokking_abmod` at
`f49969393a7482490d45483d8a4fd3046b1adede`, ported into `LabDivision` at
`c59d4426`. The scalar MLP and finite lookup construction extend the lab's
regression model at `897e80be`; the readout and loss interventions are
specified by M9.3 of [the research task](../../landscape_grokking_plan.md).
