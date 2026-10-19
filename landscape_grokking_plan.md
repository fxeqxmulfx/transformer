# Landscape explanation of modular arithmetic generalization

Started on 2026-10-19 at the user's request. Status: active, resumed at the
user's request on 2026-10-19. The original explanation and the later
number-encoder question remain unfinished. Work solo through
the task board and follow [AGENTS.md](AGENTS.md). This extends
[the Landscape framework plan](landscape_plan.md).

## Research task

Extend `Transformer.Landscape` until it explains why the model published in
the [Rotate90 Grokking Lab](https://fxeqxmulfx.github.io/grokking_abmod/)
generalizes division modulo 97, whereas an ordinary MLP taking two numbers
and returning one struggles to learn the same table, although suitable
weights can represent it. Separate the existence of a solution, access to
that solution by an optimizer, and accuracy on unseen equations. Determine
whether replacing ReLU with an analytic activation, particularly SwiGLU,
helps the proof and the training behavior.

After this explanation, investigate how to replace a fixed vocabulary of
learned vectors with an encoder of numbers while preserving useful
expressivity and trainability. State the intended domain explicitly:
all residues of a fixed modulus, unseen integers at a fixed modulus,
transfer between moduli, and arbitrary real inputs are different questions.

The proposition that gradient training cannot reach a scalar MLP solution
is a question to investigate. Specify the initialization distribution,
optimizer, step rule, sample split and time budget before making a
probabilistic or impossibility claim. A failed finite run proves neither
asymptotic failure nor an obstruction for every start.

## Model and protocol

The live site's model page and local project
`/home/misha/rust_projects/grokking_abmod` identify this architecture. Local
source revision: `f49969393a7482490d45483d8a4fd3046b1adede`; sources:
`web/own-channel/README.md`, sections Architecture and initialization and Task
and training, and `docs/dense-cuda-kernels.md`, Local residual and scalar
regression experiments. The live HTML was retrieved on 2026-10-19 and agrees
with the formula below; the deployed source revision is not asserted.

For even width W, a trainable E in R^(97 by W), a trainable A in R^(W by W),
and the fixed pairwise rotation J(u,v) = (-v,u), the lab computes

```text
N(v) = v / sqrt(1e-5 + sum_c v[c]^2 / W)
x0 = E[a] + J(E[b])
x1 = x0 + ReLU(N(x0) A)
logit[q] = sum_c N(x1)[c] E[q,c]
loss(a,b) = log(sum_q exp(logit[q])) - logit[a * inverse(b) mod 97]
```

The same E supplies both operands and the transposed readout. There are no
biases or learned normalization gains. The widths are 16, 32, 64, 128 and
256. The task includes a = 0 and excludes b = 0; it has 9,312 equations and
a seeded 50/50 train/test split. The browser trains with shuffled minibatches
and AdamW, betas (0.9, 0.98), epsilon 1e-8 and ten warmup updates. Learning
rate and weight decay depend on the selected preset.

The Rust scalar control uses two numeric features, affine/ReLU hidden
layers and an affine scalar output with MSE. Optional hidden residuals and
pre-normalization do not change the first input layer or the scalar readout.
This comparison changes the input representation, output representation,
loss, parameter sharing and sometimes depth and optimization together.
Causal conclusions require matched ablations.

## Existing evidence and proof limits

[The scalar MLP study](experiments/mod97_scalar_mlp/README.md) already
constructs a two-hidden-layer solution of width 195 and checks all equations
in Python under ReLU and GELU. Its ten training runs use the full table,
so they measure fitting rather than held-out generalization. Analytic GELU
does not make scalar division train successfully within their budget.
Every GD run eventually violates M8's sufficient curvature condition; the
sampled curvature approaches the usual quadratic stability boundary 2/eta.
Those runs do not establish a convergent spurious local minimum.

M8 proves convergence or escape under a segmentwise Hessian bound, and
almost-sure avoidance of strict saddles. It does not prove the landscape
condition for this network, identify which solutions generalize, describe
AdamW, or bound the duration of a memorization plateau. Analyticity
discharges the local Lojasiewicz hypothesis; it does not discharge these
other obligations.

M5.4 of the Landscape plan studies a clock network under hypotheses H0-H4.
It drops the zero residue, fixes embedding amplitudes, replaces the learned
branch by a character product, removes softmax competition and changes the
optimizer. These are research reductions, not established properties of
the lab's trained network. Preserve this distinction in every new result.

## Work sequence

The machine has an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM.
An elevated `nvidia-smi` query verified it on 2026-10-19 with driver
615.71.09; the sandboxed query failed to communicate with the driver.
Use the GPU for suitable training and curvature measurements, checking
PyTorch CUDA access in the permitted execution context. A sandbox failure
alone does not establish that the GPU is absent.

| Stage | Deliverable | Acceptance criterion |
| --- | --- | --- |
| M9.0 | Record the research question, exact model and existing evidence | Formulas checked against the live page and recorded local revision; task saved in English |
| M9.1 | An analytic network interface, SiLU and SwiGLU, positive-epsilon RMSNorm and finite cross-entropy | Lean proofs without `sorry`; all model parameters remain live arguments; concrete examples satisfy every hypothesis |
| M9.2 | Instantiate the tied Rotate90 residual model with analytic branches | Analyticity proved in its actual weights, with explicit readout sharing and zero residue; bridge to existing convergence results |
| M9.3 | Separate representation, fitting and generalization in matched controls | Shared splits, seeds, budgets and optimizer settings; change input encoding, readout, activation and tying separately; report numerical error as well as classification accuracy |
| M9.4 | Identify the obstruction and attraction mechanism | Quantitative gradient/Jacobian or curvature evidence produces a falsifiable statement and a Lean theorem with a nonempty regime |
| M9.5 | Explain the actual training protocol | Handle the observed step regime and AdamW explicitly; prove or test any reduction before using a phase-model conclusion |
| M10 | Encoders of numbers | Preserve a specified class of functions and demonstrate both access by training and behavior outside the training vocabulary |

Add and claim the board task for a stage before implementation. Split a
stage into smaller checked commits when its proof dependencies require it.
No theorem may assume the desired training or generalization conclusion
under another name. Only Lean results without `sorry` count as proved;
numerical measurements remain measurements.

### M9.1 results

`Transformer.Landscape.Neural.Basic` proves analyticity of SiLU, a SwiGLU
coordinate and positive-epsilon RMSNorm, including at zero activations.
`Neural.Loss` proves analyticity and nonnegativity of finite cross-entropy
and squared error. Its `crossEntropy_pos` proves that a competing class
prevents finite logits from attaining zero cross-entropy. All proofs have
concrete witnesses and no `sorry`. These are closure results; their concrete
shared-parameter instantiation follows in M9.2.

### M9.2 results

`Neural.SiLU` instantiates the complete Rotate90 network with the sole ReLU
to SiLU change. `Neural.SwiGLU` instantiates the three-projection gated
variant. Every embedding and matrix entry is a free coordinate of a single
Euclidean parameter vector, and each model reuses the same embedding in
its three roles. The residual, both positive-epsilon normalizations, every
answer class and the full cross-entropy remain. The labels and split are
arbitrary fixed data, so the results include the zero residue whenever the
data include it.

`analyticOnNhd_siluLoss` and `analyticOnNhd_gatedLoss` prove analyticity in
those actual weights. `exists_isLojasiewiczAt_siluLoss` and its gated twin
discharge the local inequality at every parameter vector using the existing
proved Landscape theorem. These results permit the analytic M8 theorems
to be instantiated; their step and curvature assumptions still require
separate evidence, and neither variant has a proved generalization result.

The parameter dimensions are proved: `|V| W + W^2` for SiLU and
`|V| W + 3 W H` for full SwiGLU. At vocabulary 97 and width 128 these are
28,800 and, with H = 43, 28,928. This is the first near-matched parameter
budget for an activation comparison; initialization, runtime and training
behavior still need to be measured.

`Transformer.Landscape.Neural.MLP` also instantiates the ordinary two-hidden-
layer SiLU perceptrons with every matrix and bias trainable. Scalar and vector
half-squared losses and full-class CE are analytic and satisfy the same local
Lojasiewicz inequality at every actual parameter vector. Their exact dimensions
match all four executable architectures. The normalized scalar half-MSE differs
from continuous residue-value MSE by a proved factor of 4,608 at modulus 97.
These common regularity properties do not explain the training gap; the
[statement review](review/mod97_mlp.md) makes that limitation explicit.

### M9.3 protocol controls

The [Rotate90 analytic controls](experiments/rotate90_analytic/README.md)
port the browser's SplitMix64 initialization, complete arithmetic split,
continuing epoch sampler and model into the lab. Independent source
fixtures match exactly at seed 11, including the width-128 checkpoint,
both splits and 21 batches across three epochs. Finite-difference tests
cover all parameters of each branch, and a separate calculation verifies
that the tied E gradient sums its three roles.
An unmodified vendored source export independently reproduces 20 complete
CPU updates on the current runtime, with exact losses, accuracies and final
weights. Detached observers retain the full nonconstant character spectrum,
participation rank and overlap of the two centered operand subspaces.

The prepared CUDA experiment pairs seeds 11, 17 and 23 across ReLU, SiLU
and SwiGLU (H = 43), with the browser's W128 Fast optimizer settings and a
1,000-epoch budget. The lab's locked PyTorch CUDA backend is an explicit
deviation from the native/browser kernels. All nine runs completed with
100% final train and held-out accuracy. First held-out 99% takes 29/31/30
epochs for ReLU, 28/26/27 for SiLU and 14/15/16 for SwiGLU. The gated runs
later dip to 94.05%, 95.25% and 86.10%, with corresponding training dips;
the dense controls never fall below 99% after their first crossing.
Eight energy-selected character pairs retain 100% held-out accuracy in all
nine final checkpoints. The [experiment results](experiments/rotate90_analytic/README.md)
contain every seed, full curves, projection ablations and provenance.
An actual gradient-attraction mechanism remains open. The
[matched MLP controls](experiments/mod97_representation_controls/README.md)
now separate scalar/one-hot operands, scalar/vector answers, vector MSE/CE
and decay 0/1 across the same three source splits. All have two SiLU hidden
layers of width 195. The parameter counts and input fan-in changes are
explicit; these are not matched-capacity controls. Tests construct finite
scalar SiLU weights that classify every equation, independently of the
random starts used for training. At the user's stop, 11 of 36 runs completed,
one retained its 25,000-update checkpoint, and 24 had not started. The
[partial snapshot](experiments/mod97_representation_controls/progress.json)
preserves this incomplete grid; it is not a final comparison result.

The resumed workspace does not contain the ignored raw runs behind that
snapshot. Its available device is an NVIDIA GeForce GTX 1050 (2 GiB),
verified with `nvidia-smi` and the locked PyTorch runtime on 2026-10-19;
the earlier RTX 3050 observation describes the previous environment.
Board task #74 completed the entire unchanged 36-case grid afresh on one runtime
and device. Preserve `progress.json` as the earlier partial cohort and
do not combine its measurements with the new trajectories. Every new run
reached its original 30,000-update budget and passed exact final-checkpoint
replay, finite-record, split, source and initialization audits. All three
one-hot/vector MSE runs without decay finish at 100% on both splits;
changing either code alone loses this outcome. One-hot/vector CE without
decay memorizes all three training splits and fails on held-out equations.
With decay one, CE succeeds only at seed 17 and MSE succeeds in no seed.
These qualified finite results and every numeric error are retained in the
complete findings. They do not yet identify the actual attraction mechanism.

Task #75 prepares a full-table Jacobian diagnostic for all twelve scalar
readouts, at their original initialization and final trained coordinates.
It measures the target and residual contractions after projecting away
additive operand effects, with all zero-numerator equations retained.
Independent parameter derivatives and finite differences check the
diagnostic. These are local Euclidean tangent measurements on CPU float64,
not an AdamW reduction or a claimed gradient-attraction theorem.

The [SwiGLU stability study](experiments/rotate90_swiglu_stability/README.md)
tests rates 0.01, 0.003 and 0.001 on the same three seeds. It retains a
common 10,000-update prefix and extends the smaller-rate budgets to nearly
equal cumulative learning-rate times decay. Per-tensor update and moment
observations help distinguish optimizer motion from held-out-only failures.
The nominal clock is an experimental comparison, not an assumed gradient
flow or exact AdamW orbit equivalence.
All nine controls completed. The 0.01 baselines reproduce the earlier
model tensors exactly, including their later loss of fit. Rates 0.003 and
0.001 have no recorded post-crossing falls below 99% in any seed, either
at the common 10,000-update prefix or through their longer matched-time
budgets. Their first 99% crossings take 30/35/30 and 80/92/75 epochs.
The [portable results](experiments/rotate90_swiglu_stability/results.json)
and full curves preserve both finite horizons and every seed; this is not
an asymptotic stability or gradient-attraction theorem.

### M9.4 interaction obstruction

`Transformer.Landscape.Neural.Interaction` proves that the actual normalized
tied readout without its residual branch cannot separate a division
checkerboard, for any shared embedding. The concrete modulus-97 theorem
keeps width 128, epsilon 1e-5 and the smallest-index tie policy's necessary
comparisons. An additional bridge identifies this readout at a zero dense
matrix of the complete SiLU parameterization. This supplies a necessary
operand interaction, not an attraction theorem or a scalar-MLP barrier.
See [the statement and source review](review/rotate90_interaction.md).

`Neural.Products` proves that opposite SiLU gates realize exact products
and compiles sums of those products into the actual SwiGLU parameter vector,
including both normalizations, residual and tied logits. A separate
[constructive checkpoint](experiments/rotate90_analytic/expressivity.py) uses
nine products and 18 of H=43 individual gates to compute a faithful quotient
character and retain zero numerators. Every one of the 9,312 equations is
classified correctly in CPU float64, CPU float32 and CUDA float32, with
all parameters bounded by one. The full character-checkpoint correctness
is numerically checked; the paired-gate compiler is proved in Lean. This
also puts a correct classifier inside the known AdamW parameter container.
It supplies a representable structured solution, not its random-start
attraction mechanism; see [the source review](review/rotate90_products.md).

### M9.4 tangent control

Tasks #75 and #76 measure and formalize the explicitly frozen MLP control.
All twelve fresh scalar-output runs now have initial and final full-table
Jacobians, centered target/residual contractions, and 512-row frozen-kernel
spectra and GD trajectories in the [access results](experiments/mod97_representation_controls/access_results.json).
One-hot inputs start with smaller target Rayleigh quotients than scalar
inputs, but without decay develop much larger sensitivity during training.
Their initial frozen kernels interpolate the 512 sampled training equations
at the long numerical horizon while retaining large held-out numeric error.
The scalar initial kernels have only 30–32 retained numerical directions
at the stated relative cutoff. These are measurements of conditioning,
feature changes and sampled interpolation, not exact rank or attraction claims.

`Neural.Tangent` differentiates the actual frozen half-squared objective,
derives its normalized `J J^T / n` kernel and proves its GD eigenmode
recurrence. A unit mode with `eta * mu <= 1` and `k * eta * mu <= 1/2`
forces loss at least its initial squared component divided by `8 n`.
The nonempty witness has `J x = x/10`, positive eigenvalue `1/100` and a
zero-loss point, yet loss at least `1/8` after 50 GD steps of size one.
An exact null mode also gives a permanent frozen-model error floor;
numerical cutoff directions are not certified as exact null modes.
The [statement review](review/mod97_tangent.md) separates this theorem from
the changing neural features, original minibatches and AdamW. The actual
Rotate90 attraction and generalization mechanism remains open.

### M9.5 exact-real optimizer container

`Transformer.Landscape.Neural.AdamW` matches the inclusive warmup and
AdamW betas, optimizer epsilon and decay of the experiments. For every
base rate between zero and one, initialized native buffers and the actual
recurrence imply a coordinate ceiling `max(abs(initial), 1.7)`, with no
future gradient bound. The computed box is a compact Euclidean parameter
container at the finite indices of both analytic networks. This discharges
a boundedness obligation for that exact-real recurrence; AdamW convergence,
stochastic attraction and the FP32 bridge remain open.
The [optimizer statement review](review/rotate90_adamw.md) records its
nonconstant initialization witness, topology and verification outcomes.

## Activation comparisons

SiLU is a direct scalar replacement, s(t) = t / (1 + exp(-t)). Standard
SwiGLU is a gated branch, for example

```text
branch(r) = (s(r G) elementwise_mul (r U)) D
x1 = x0 + branch(N(x0))
```

It introduces separate gate and value projections and an optional output
projection. Report the intermediate width, parameter count, initialization
and FLOPs. Replacing each ReLU coordinate with SiLU and adding a SwiGLU
branch are distinct interventions. Positive normalization epsilon and the
positive softmax denominator make both variants analytic in finite real
parameters. The original ReLU network requires a separate treatment of
activation boundaries, even if automatic differentiation returns a Hessian
away from them.

## Mechanisms to test

For squared error, the local prediction Jacobian J gives the first-order
residual update through J J^T. Measure how much of the target residual lies
in directions with small eigenvalues, and compare scalar and learned-vector
encodings. A poorly aligned Jacobian can explain slow access to an
expressible solution without asserting a bad local minimum. Track changes
of the Jacobian during feature learning before drawing an asymptotic
conclusion from its initial value.

For the classifier, examine the learned embedding and readout geometry,
Fourier structure on nonzero residues, the zero-residue pathway, margins,
parameter norms, competing answer logits and optimizer state. Identify a
quantity that predicts held-out equations rather than only lower train
loss. Compare its development before and after the observed accuracy rise.

Use plain GD as a theorem-aligned control and the browser's AdamW as the
target protocol. Decoupled AdamW decay is not automatically the Euclidean
gradient of cross-entropy plus an L2 penalty. Unregularized separable
cross-entropy can have its infimum only at diverging norms, so convergence
to a finite global minimizer must not be silently assumed.

## Completion

The first question is answered only when a precise, validated mechanism
connects the published network's representation and optimizer to held-out
success, and accounts for the scalar control's observed failure under a
specified protocol. An analytic replacement, interpolation construction or
generic convergence theorem alone is intermediate progress. Start the
second question after establishing that mechanism, while keeping its
required domain and expressivity criteria recorded here.
