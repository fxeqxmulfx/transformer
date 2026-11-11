# Convex channels with a flat-spectrum capacity bound

Can a jointly convex model learn both routing and generated values from ordinary
answer cross-entropy? The construction in Lean commit `dbb3a4cb` replaces fixed
copying by a direct tensor `W[query,key,input_value,output_class]`. Its nonnegative
entries have output marginals independent of input value and total routing mass
one. The forward is `z(c)=sum_k W[q,k,D(k),c]`. No separately multiplied learned
matrices, arithmetic operations or task rules appear in the forward.

`review/flat_spectrum.md`, CC1–CC8, proves joint convexity, expressivity beyond
copying, identification from separating answer examples, an exact linear
optimization oracle and the applicable flat-spectrum capacity bound. It also
proves that two flat full-rank targets can remain indistinguishable on a partial
training set. Capacity and identification are separate obligations.

## Frozen protocol

Fixed before the research runs on 2026-11-11. All 18 runs are required; report
every seed at its declared endpoint, without test-selected extensions or tuning.
The five implementation checks use a separate seed or exhaustive tiny examples.

| Item | Protocol |
| --- | --- |
| Seeds | 0, 1, 2; identical train/test examples for every model on a task/seed |
| Inputs | Four keys, eight categorical values; four shuffled adjacent key/value pairs followed by all four independently shuffled queries |
| Data | 256 training dictionaries and 256 disjoint held-out dictionaries, from 4096 possible dictionaries |
| Outputs | Full 20-class vocabulary: four keys, eight input values, eight additional output symbols |
| Ordinary MQAR | Answer is the value bound to the query |
| Recoded MQAR | Each query has an independently shuffled value-to-output permutation; outputs use the eight additional symbols, absent from every input |
| New channel | 2560 joint parameters, uniform feasible initialization; full-batch Frank-Wolfe, 1000 updates, `eta=2/(step+1)` starting at step one |
| Fixed-copy control | 16 matching parameters, uniform initialization; full-batch projected GD, learning rate one, 1000 updates |
| Conventional transformer | Width 32, two layers, four heads, ReLU FFN width 128, post-LayerNorm residuals, sinusoidal positions, learned embeddings and untied readout |
| Transformer training | AdamW, learning rate .003, decay .1, beta .9/.98, epsilon 1e-8, ten-step warmup; minibatch 32, 6000 updates |
| Loss | Ordinary answer CE over all 20 classes; supervision only at the four query positions |
| Observations | Steps 0,1,2,5,10,20,50,100,200,500,1000; transformer also 2000,4000,6000 |
| Execution | Native Rust; validated Kernite CPU transformer from `convex_recall_compare`, four workers per transformer, three concurrent seed streams |

The new task is smaller than the earlier eight-key comparison; do not combine
their accuracies or timing measurements. Convex methods see 1024 answer labels
per update and the transformer 128, with different total update budgets. These
are learning and representation comparisons, not equal-FLOP or speedup claims.
The transformer uses f32 parameters/backpropagation; convex parameters and all
reported CE use f64. Kernite is pinned to clean commit
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c` in the existing local checkout.

The label permutation is constructed only by the dataset generator. Channel
forward reads the supplied query and categorical bindings. A test reconstructs
those bindings from visible tokens, checks disjoint train/test dictionaries and
verifies that no recoded answer occurs in the input. The runner checks the exact
CC5 atom-separation premise on each training dataset. Passing this premise is a
data property; it is not imposed by answer-dependent forward features.

The exact CC8 oracle first chooses the cheapest class at each key/value, then
the cheapest key for each query. It receives only the ordinary CE gradient.
Updates are convex combinations with that vertex, preserving feasibility.
The report includes feasibility error, distance to the realizable rule and the
full-data Frank-Wolfe gap. The latter is a floating-point optimization diagnostic,
not a machine-checked guarantee about numerical rounding or an optimizer rate.

## Spectral diagnostic

Only for evaluation, read the eight answer symbols as four `+1` and four `-1`
values. Enumerate all 4096 dictionaries. For both tasks, the target's `4 x 4096`
Gram is exactly `4096 I`: rank four, squared singular value 4096. The checker
measures errors of explicit rank-`d` row truncations for `d=0,...,4`. They must
equal `4096(4-d)`, saturating the proved R3 lower bound. This diagnostic is never
an input feature, a training label supplement, or an optimization penalty.

The channel's actual scalar output factors through 32 categorical binding
features. That upper feature count is sufficient for the rank-four target and
the R3 bound gives no positive error floor at width 32. It is not a minimal
width claim. A separate full-domain Frobenius error is reported for the learned
channel. Classification accuracy and this scalar regression error are different
metrics. Flatness alone is not claimed to imply learnability of unseen labels;
the two-table ambiguity from Lean is also independently checked natively.

The copy-only control has exactly zero possible accuracy on recoded MQAR because
its nonzero logits are confined to input-value classes. A successful new model
therefore demonstrates learned output generation, not a more confident copy.
Channel logits sum to one, so its ideal CE is `log(e+19)-1`, not zero. Comparing
its CE directly with an unbounded-logit transformer's confidence is misleading.

## Reproduce

No Python is used. From the repository root:

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
cargo test --offline --manifest-path experiments/convex_channels/Cargo.toml
cargo fmt --manifest-path experiments/convex_channels/Cargo.toml -- --check
cargo build --offline --release --manifest-path experiments/convex_channels/Cargo.toml
experiments/convex_channels/target/release/suite experiments/convex_channels/runs
experiments/convex_channels/target/release/report experiments/convex_channels/runs > experiments/convex_channels/runs/summary.json
```

The suite refuses to overwrite an existing run. It saves input/source hashes,
configuration, every observation, final parameters and optimizer moments, and
stdout for every seed. It has no resume mode; an interrupted run remains as
incomplete evidence. Use a fresh directory for a separately documented rerun.
The report requires all 18 completed endpoints and validates source fingerprints,
regenerated data, spectral diagnostics and the recorded endpoint consistency. It
reconstructs every saved channel and re-evaluates its predictions, gradient gap,
constraints, parameter witness and full-domain spectrum.
Generated evidence stays ignored under `runs/`, linked to the root checkout
while working in a temporary worktree.

## Findings

All 18 frozen runs completed and passed report validation. No failed runs,
restarts, changed endpoints or test-selected reruns occurred. Every model's
training/test input hashes agree within its task and seed. All runs have source
fingerprint `0a2d91eea63b1754ed0c0f2208f4d6d27de4580162c406aaebcba88b734e20fc`.

| Model | Parameters | MQAR test accuracy, all seeds | Recoded MQAR test accuracy, all seeds |
| --- | ---: | ---: | ---: |
| Joint convex channel | 2560 | 100% | 100% |
| Fixed-copy matcher | 16 | 100% | 0% |
| Conventional transformer | 26112 | 100% | 100% |

The channel first reaches 100% at update **one**, in all six runs. The stored
tensor equals the entire realizable parameter witness exactly: maximum parameter
error, constraint violation and full-data Frank-Wolfe gap are all zero. Replaying
the saved weights reproduces these diagnostics. Its scalar output matches the
flat target on **all 4096 dictionaries**, not only the 256 held-out examples:
full-domain squared Frobenius error is zero. Ideal CE is 2.0781543864329897,
consistent with the fixed unit-logit scale.

This one-step behavior has a simple explanation for this dataset: at uniform
initial logits, the CE gradient counts observed labels for each query/key/value.
The queried key predicts all its labels consistently; separating examples
contradict alternative keys. The exact finite oracle therefore selects the
correct key and its learned output classes. The first step has `eta=1`, so it
reaches that vertex. This explains these measured runs; the Lean learning theorem
is about global optimizers and the checked data condition, and the Lean oracle
theorem does not formalize this complete one-step numerical argument.

On ordinary MQAR the fixed-copy matcher first shows 100% at update two for every
seed. The transformer's first observed 100% is at updates 500/200/200 on ordinary
MQAR and 500/1000/1000 on recoded MQAR. Endpoint training accuracy also equals the
test accuracies in the table. Mean transformer test CE is `8.23e-8` and `6.15e-5`
respectively; copying's mean recoded CE is 3.05377. Different confidence scales,
batch sizes and update budgets prevent a direct speed or CE-quality claim.

For every task and seed, the target Gram is `4096 I`. Measured errors at ranks
0,1,2,3,4 are respectively 16384,12288,8192,4096,0, exactly the flat-spectrum bound.
The channel retains ample categorical capacity and successfully learns generated
outputs. The result relies on its shared query/key/value structure and observed
coverage. It does **not** establish that arbitrary unseen labels are determined
by flatness, or resolve the compact learned-embedding transformer replacement.

Validation: five native tests, formatting, all 18 endpoint checks and saved-channel
replays pass. Lean's full build and 62-declaration channel axiom audit pass; the
repository retains its existing 36 unfinished statements, with zero results
resting on them and zero extra axioms. Generated evidence is in the ignored root
`experiments/convex_channels/runs/` and `.kanban/artifacts/144/` directories.
