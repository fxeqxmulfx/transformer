# Nonlinear causal paths with a convex output tensor

Can deeper generic causal features repair the remaining depth and recall
failures while ordinary cross-entropy stays convex in the learned coordinates?
Lean comes first: CD5-CD7 preserve complete nonlinear paths; CD8 at `32a9b668`
allows a learned combined path/feature/class tensor under per-class absolute-sum
budgets. The blocks remain fixed. Separate learned routing and decoder factors
would introduce a product and are not covered by that joint-convexity claim.

This directory contains the path bank, a learned output tensor, an allocation
preflight and training-only optimizer checks. No held-out stress accuracy is
claimed. Independent-routing and additive components are implemented; the
complete comparison runner and protocol must be frozen before scoring held-out
splits. The earlier training-only results below remain tied to their producer.
The interrupted earlier 35-task cohort remains unchanged.

| Component | Fixed construction | Information used |
| --- | --- | --- |
| Input views | All symbols, plus every pair of the four most frequent training symbols | Training input tokens only |
| Previous-distinct block | Move the current complex position code to the latest earlier distinct symbol in its view | Visible prefix and previous path state |
| Completed-successor block | Find the latest completed occurrence of the selected symbol and move to its next visible symbol | Visible prefix and previous path state |
| Stream transport | Identity or fixed phase-conjugated normalized Fourier-4 | Layer index only |
| Terminal features | One-hot final symbol and fixed FNV history bucket | Actual executed path history |

Positions are represented by four complex amplitudes at each prefix location,
including an explicit missing-position slot. Routing reads stream energy from
the actual preceding hidden state. A branch clips that state's norm at one,
moves the selected content, and transports its displacement. Its norm is at
most two. The full block computes the transported skip plus that bounded
displacement; the numerical implementation checks its norm on every step.
No block reads a target, task identity or arithmetic operator.

The current evaluator stores only nonzero complex position rows, retaining even
small roundoff residuals. It still executes the complex skip and branch update;
it does not replace the hidden state with a symbolic pointer. A frozen dense
evaluator from `a440f6a0` checks exact feature-ID agreement across both
transports, depths 0–4 and prefix lengths 1–128. This bounds temporary state
storage by the path's visited positions instead of allocating a dense prefix
matrix at each branch. The norm check applies to the complex position code;
the history is auxiliary metadata and its one-hot terminal features are directly
bounded by one.

The identity/Fourier pair is a coordinate-invariance control. Both the skip
and the branch displacement are transported, and position readout uses energy;
their exact predictions should agree. This is not the same experiment as
changing only a skip while holding a coordinate-sensitive branch fixed, and
does not claim a Fourier accuracy advantage. The unitary skip is the flat
spectral bound. A bounded feature passed to the CD8 head has a logit bound
from its coefficient budget; no whole-network Jacobian flatness is asserted.

There are at most seven views and `2^depth` paths per view. Depth is capped at
four, prefixes at 1024 tokens, vocabulary at 1024, and history slots at 4096.
The preflight uses the original frozen easy/hard depth, long MQAR and short
binding datasets, checks their hashes through the existing native constructor,
and estimates coefficient, gradient, projection scratch, feature-cache and
observation-index allocations for
depths 0, 1, 2 and 4. Explicit buffers above 2 GiB must be rejected before
allocation. An 8 GiB OS address-space limit must also cover data, libraries,
temporary buffers and serialization. Hash collisions remain part of the model.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexDepth
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexDepthAxioms.lean
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress_depth/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_stress_depth/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress_depth/Cargo.toml
bash experiments/convex_stress_hull/limited.sh 8 experiments/convex_stress_depth/target/release/convex-stress-depth prepare
```

Twenty native checks cover the complex path invariants, early allocation
rejection, independence from labels/task names, CE finite differences, exact
L1 projection, example minibatches with repetitions and tails, empirical
forward frequencies, an exhaustive linear-oracle comparison, and checkpoint
roundtrip/corruption handling. They also check sparse/dense feature equivalence,
additive independence from intermediate states, the independent factor gradients,
budget-preserving embedding, vertex initialization and factor serialization.
These are implementation checks, not benchmark
outcomes or a learning convergence proof.

The independent baseline uses one Bernoulli probability for each block layer
and a shared learned decoder over input-view/terminal-feature coordinates.
Its full-path weight is the product of layer probabilities. CD5's
`productWeights_mem` and CD8's `liftHead_budget` at `6f71b458` embed it in the
same radius-R combined tensor domain. Optimizing the factors is nonconvex.
It can start from the exact same single-path forward proposal as the joint head.
An update projects the decoder's CE step, then computes the routing gradient
at that decoder and uses at most eight forward backtracking checks, starting
at step 0.1. Every backward pass and forward check must be charged to its run.

The additive baseline evaluates every layer's two blocks on the original state
and learns one combined bounded readout. Its depth-one features equal the
composed depth-one features. Further layers repeat those shallow feature
functions in this energy-invariant library; they do not acquire interactions
between successive states. This is a deliberate capacity control, with its
larger stored parameter count reported rather than interpreted as extra depth.

The initial allocation preflight at `ba53e536` completed in 2.26 seconds
at 244.5 MiB peak RSS under the 8 GiB process limit. Depth-4 MQAR would require
4,714,141,320 bytes with the expanded scratch/index accounting and is rejected
by the 2 GiB budget;
its depths 0, 1 and 2 fit. All four depth choices fit for the two depth tasks
and short binding. These are allocation estimates, not measured training peaks.
The preflight, resource log and component-check logs are retained outside the
worktree in `.kanban/artifacts/175/`.

`train_check.json` fixes a training-only diagnostic with radius `32 * vocab`,
depths 0/1/2/4, and the original data hashes and supervision multiplicities.
Every mode first builds the same training-label frequency proposals, with
log frequencies floored at -24, then selects a feasible proposal by original
training CE. Full-batch and minibatch refine that initializer by projected CE
gradient steps; forward-only constructs no loss gradient. These are not
comparisons of optimizers starting from zero. The initializer's complete data
and time costs are recorded for all three modes.

Full-batch has at most 16 updates and minibatch at most 256, using the original
example batch size, seed-1 shuffle and final short batch. At least one gradient
update precedes the `1e-8` training-CE stopping check. The learning rate is the
reciprocal of the active-feature count. Full gradients produce a numerical
linear-oracle gap; forward-only leaves that field null. This diagnostic does
not claim a proved finite-iteration convergence rate.

The serial launcher enforces an 8 GiB address-space cap on both training and
replay. Checkpoints stream coefficients as binary floats without a large JSON
copy. Each result records the compiled source fingerprint, producer commit,
data fingerprints, model hash, costs and gradient/example/prefix counts.
Replay reloads the saved model, reconstructs training-only feature discovery,
recomputes original training CE and checks the class budgets.

```sh
export DEPTH_PRODUCER_COMMIT=$(git rev-parse HEAD)
# The parent of FRESH_DIRECTORY must exist; existing runs are never overwritten.
bash experiments/convex_stress_depth/run_train_checks.sh FRESH_DIRECTORY gpt-probe-key-value-binding 1
```

No validation or test metric is computed by this command. The larger depth
comparison and fresh matched transformer baseline remain outstanding.

The first check is complete at producer `a440f6a0`, source fingerprint
`21355e87dcc3e1b77e4524e075d4591c9a967a2b3afb478c2348add827147cbf`.
It used all 2,048 original short-binding training examples at depth one.
All three modes selected the same forward initializer with training CE
`3.7373833906779454e-9`; refinement left that reported CE unchanged.

| Mode | Gradient updates / prefixes | Peak training RSS | Total training wall time |
| --- | ---: | ---: | ---: |
| Full batch | 1 / 2,048 | 101,512 KiB | 0.68 s |
| Minibatch | 1 / 32 | 101,552 KiB | 0.66 s |
| Forward only | 0 / 0 | 101,544 KiB | 0.62 s |

Each row additionally pays for the shared initializer's 2,048 examples and
prefixes. Times include data, feature construction, initialization, refinement,
the final diagnostic and checkpoint writing; separate replay took 0.46–0.49 s.
All three binary models replayed with CE difference `3.46e-16` or less and
passed the class-budget check. Their largest class absolute sum is about 24,
below the radius 3,200. The full-gradient linear-oracle gaps remain about
`2.00e-5`; no tighter gradient-gap result is claimed from the small CE itself.
No held-out accuracy was measured, and the one-step gradient refinements did
not discover the good initializer.

The real depth-4 MQAR allocation guard also rejected its 4,714,141,320-byte
buffer request before constructing the cache or head, using 244,012 KiB peak
RSS for data verification. A deliberately corrupted saved CE was rejected on
replay. Models and resource logs live in
`runs/training-check-v1-binding-d1/`, linked to the root checkout; the frozen
producer, checks and negative controls are retained in board artifact #175.
