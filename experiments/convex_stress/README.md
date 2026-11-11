# Convex channels on the GPT short and long stress tests

Can the general convex channel and the direct forward-only learner transfer
from the paired-dictionary experiment to the existing GPT stress tests?
The benchmark is in `/home/misha/python_projects/diff-gpt`, at source commit
`3555299`, under `research/complex_geometry_suite/`. The originally suggested
`python_projects/gpt` path is absent on this machine.

The complete protocol is frozen in `protocol.json` before the scientific
runs. Both arms receive identical causal inputs and ordinary answer labels.
There are 25 short and 10 long tasks, three fixed seeds per arm: 210 cells.

| Item | Frank-Wolfe channel | Forward-only channel |
| --- | --- | --- |
| Parameters | Direct joint routing/output tensor, sparse storage | One selected key and categorical decoder per query |
| Training | Ordinary full-vocabulary CE, exact linear oracle, convex updates | Record labels and eliminate conflicting keys; no gradients |
| Initialization/order | Uniform logits; integer first oracle; smallest-index ties | Empty records; increasing candidate key order; default class zero |
| Budget | Complete data passes within the source example budget | One complete finite candidate search, at most keys times observations |
| Completion | First observed 99% primary accuracy, optimum certificate, or fixed budget | First consistent key per query, or explicit failure |
| Unsupported fit | Continue optimizing the whole convex channel family | `no_consistent_key`; no invented fallback accuracy |

## Data and input adapter

The five files in `src/vendor/` and `suite.json` are exact copies of the
frozen native source; `SOURCES.sha256` records their hashes. They reproduce
the original dataset fingerprints without running Python. Task rules appear
only in these unchanged generators, never in the learned forward map.

The channel query is the current visible token. Its categorical dictionary
contains four fixed feature banks: relative-lag tokens, the latest completed
successor at distance one for each token, the latest completed successor at
distance two, and token presence. Missing values are PAD; presence has one
additional categorical symbol. Context capacity is the largest declared
input/rollout length among the benchmark splits. Only this length metadata,
not held-out answers or their correlations, sets the input dimensions.
The same feature definition is used for every task. It supplies no modular
arithmetic features, parity answer, task identifier or target rule.

This is an explicit causal sequence adapter for the channel model, whose
earlier MQAR experiment used already parsed dictionaries. The successor
banks expose generic associations; they are part of the inductive bias and
must be included when interpreting recall results. The adapter is fixed,
so all logits remain linear in the trained joint coefficients. At a fixed
query the categorical feature count is `keys * values`. CC6's rank bound
applies to kernels whose dictionary columns are held fixed across query
rows. The causal adapter uses the whole prefix; we do not assume that each
task's conventional two-input matrix has that same factorization. No
nontrivial spectral error floor is measured for this suite. Capacity bounds
and identification of missing answers remain separate requirements.

## Optimization and evidence

At uniform logits, each linear-oracle decoder chooses the most frequent
label for each `(query,key,value)`; the best key maximizes the sum of those
counts. This gives the exact first Frank-Wolfe vertex without a dense tensor.
If it realizes a query's entire training row, it attains the proved global
CE lower bound, so that query can stop with an exact certificate. Otherwise
full-batch CE gradients and `eta=2/(iteration+1)` updates continue. Every
unobserved value uses the class-zero completion of the oracle. Classes never
seen as labels remain in the softmax denominator; output vocabulary is not
restricted. Sparse model rows retain the shared routing-mass equalities.

The forward-only implementation ports CF1-CF5 and the native learner from
`experiments/convex_channel_forward` at `3e0e2cb1`, changing only dictionary
storage to sparse PAD rows. It keeps full candidate-scan visit counts.
Unobserved queries return class zero, as the source's empty-row learner does;
the iterative arm retains uniform logits there. Both have argmax zero.
An inconsistent query causes an explicit whole-model fit failure, with the
failed queries and search counts retained. Successful fits are global CE
minima at the same bounded unit-logit scale.

The full-batch iteration budget is
`floor(max_updates / ceil(train_examples / batch))`, with any partial last
epoch unused. Seeds shuffle the training summation order once using the
source RNG and `seed+10000`; they do not change initialization or data.
The direct learner is order-independent on successful consistent data.
Primary evaluation occurs after each full data pass. This is coarser than
the source's intra-epoch observation schedule on some tasks. Consequently
these runs are a transfer test with documented optimizer differences, not
a matched transformer speed comparison or a replacement for its closed
promotion cohort. No validation-dependent feature changes, extensions of
failed budgets or seed selection are permitted.

Metrics preserve the original full-split definitions: complete exact
sequences, autoregressive generation reading its own outputs until EOS or
the fixed limit, recurrence residue-probability pooling, and auxiliary
numeric error/coverage. CPU f64 arithmetic replaces CUDA f32. Every source
split, including longer and shifted variants, is scored at the endpoint.
Saved models are replayed to verify every metric. There is no held-out
evaluation for an explicit forward fit failure because no model was returned.

The observation theorem supplies a sufficient condition for predicting
unseen inputs from observed features. The iterative arm also records an
input-only necessary test for that certificate: a test coordinate never
seen at its query rules out that feature-span reconstruction. Complete
coordinate coverage alone is **not** a span certificate. This diagnostic
uses the first answer prefix for generation, and all supervised prefixes
otherwise. A failed certificate is not proof that all identification
methods fail; the nonnegative channel constraints can be stronger.

## Reproduce

```sh
cargo test --offline --manifest-path experiments/convex_stress/Cargo.toml
cargo fmt --manifest-path experiments/convex_stress/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path experiments/convex_stress/Cargo.toml -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_stress/Cargo.toml
bash experiments/convex_stress/run_suite.sh experiments/convex_stress/runs/stress-v1 4
```

The runner requires a fresh cohort directory and preserves each cell's
stdout, exit status, metadata, dataset/source hashes, learned coefficients,
metrics and replay log. Generated outputs stay ignored under `runs/`, linked
to the root checkout while working in a worktree. Native tests check causal
features, finite-difference gradients against a separate dense forward,
the full-vocabulary oracle by exhaustive vertex search, shared-mass
feasibility, saved predictions, forward consistency on all 81 partial
Boolean datasets, teacher forcing isolation, greedy rollout, residue
pooling and numeric penalties.

## Findings

All **210 cells completed**, including 18 explicit forward fit failures.
There were no process failures, restarts, extra seeds or extended budgets.
All dataset fingerprints matched, and all 210 replay checks passed: saved
models reproduce every split's metrics; explicit failures reproduce the
same failed query rows and search counts. Measured source commit `7d7bd1e6`
has embedded SHA-256
`4452feddd3eeb5cc82464029b2f733c4fd7209bf8a79156fe907ed3d3c19796e`.
Evidence is preserved in `runs/stress-v1/` and `.kanban/artifacts/147/`.

| Learner | Short: primary exact sequences >=99% | Long: primary exact sequences >=99% |
| --- | ---: | ---: |
| Frank-Wolfe | 9 / 75 | 0 / 30 |
| Forward-only | 9 / 75 | 0 / 30 |

The three successful tasks are the two Boolean AND tasks and linear
transport, each at 100% in all seeds after the first full data pass. Primary
accuracies are identical across all three seeds for each task/arm. This
agreement tests summation order, not different data or random initializations.
The Boolean primary novel splits contain only seven and eight examples;
their larger auxiliary splits remain in the saved endpoint reports.

The full task accounting below gives primary exact-sequence percentages.
Each entry applies to all three seeds. **No fit** means that the direct
learner returned no model; it is a failed benchmark cell, not a measured
zero-percent classifier. Arithmetic rows are retained because they belong
to the requested frozen suite, not because arithmetic features were added.

| Tier | Task | Frank-Wolfe, % | Forward-only, % |
| --- | --- | ---: | ---: |
| Short | suite-boolean-and-shift | 100 | 100 |
| Short | suite-boolean-and-random | 100 | 100 |
| Short | power-division-p97-k3 | 0.9450 | 0.9450 |
| Short | arithmetic-cube-plus | 1.0840 | 1.0840 |
| Short | power-division-p193-k5 | 0.4642 | 0.4642 |
| Short | power-division-p193-k2 | 0.4642 | 0.4642 |
| Short | power-division-p97-k2 | 0.9450 | 0.9450 |
| Short | arithmetic-square-times | 2.0191 | 2.0191 |
| Short | arithmetic-sub | 0.8927 | 0.8927 |
| Short | arithmetic-mul | 2.0191 | 2.0191 |
| Short | arithmetic-add | 0.9564 | 0.9564 |
| Short | gpt-probe-key-value-binding | 86.1328 | 86.1328 |
| Short | power-division-p193-k3 | 0.4642 | 0.4642 |
| Short | division-p193 | 0.4642 | 0.4642 |
| Short | basis-depth-easy | 42.9688 | No fit |
| Short | arithmetic-square-sum | 1.9554 | 1.9554 |
| Short | gpt-probe-branch-amplitudes | 0 | No fit |
| Short | power-division-p97-k5 | 0.9450 | 0.9450 |
| Short | arithmetic-half-sum | 0.9564 | 0.9564 |
| Short | arithmetic-sum-square | 0.9564 | 0.9564 |
| Short | basis-depth-hard | 20.3125 | No fit |
| Short | arithmetic-affine-product | 2.3804 | 2.3804 |
| Short | arithmetic-perm-add | 1.0202 | 1.0202 |
| Short | gpt-probe-linear-transport | 100 | 100 |
| Short | arithmetic-div | 0.9450 | 0.9450 |
| Long | arithmetic-abs-diff-blind | 2.8905 | 2.8905 |
| Long | arithmetic-abs-diff | 0.8927 | 0.8927 |
| Long | arithmetic-sum-pow4 | 0.9564 | 0.9564 |
| Long | basis-parity20 | 46.8310 | No fit |
| Long | basis-parity16 | 54.1401 | No fit |
| Long | arithmetic-sum-cube | 0.9564 | 0.9564 |
| Long | basis-recall-easy | 50.0000 | 50.0000 |
| Long | recurrence-fibonacci-k1-c2-lgt | 1.1267 | 1.1267 |
| Long | associative-divisor-n2-l1-w0 | 53.8660 | 53.8660 |
| Long | basis-parity24 | 50.0000 | No fit |

There are 87 exactly realizable cells per learner: both reach 100% training
accuracy and the global ordinary-CE lower bound there, but only nine pass
the held-out threshold. The other 18 direct cells fail consistency: depth
easy/hard, branch amplitudes, and parity16/20/24. Frank-Wolfe continues on
these data. Depth-hard and parity16/20 stop at the documented numerical
FW-gap tolerance after four/two passes; these are floating-point
optimality diagnostics, not exact Lean certificates. Depth-easy, branch
amplitudes and parity24 exhaust 16, 937 and 80 full passes respectively.
Their seed-zero mean FW gaps are approximately `4.33e-4`, `8.92e-6`, and
`1.65e-5`; they are not reported as converged. The maximum measured channel
constraint violation across returned models is `1.12e-15`.

Recall exposes a concrete sharing problem. Both arms fit their entire
training sets. Key/value binding reaches 86.13% on 512 held-out queries;
long recall reaches 91.82% token accuracy but only 50% complete sequences
of eight answers. Its auxiliary test split reaches 47.27% exact sequences.
The post-run `check_recall.jq` check verifies that every learned query routes
to its own visible successor and that every stored token-value row copies
the correct value. However, decoders are separate per query: only 888 of
1024 possible query/value records occur in short recall, and 59871 of
65536 occur in long recall. Unobserved records output the frozen default
class zero. Thus correct routing and exact training fit leave a substantial
gap on new query/value combinations. These counts explain why the earlier
small, fully covered MQAR experiment did not predict this suite's outcome.

The feature-coverage diagnostic also rules out the O2 span certificate on
every held-out recall input. This is a sufficient-certificate limitation,
not an impossibility theorem for recall: a representation that shares
value decoding across equivalent queries could use other observations.
Such sharing can be imposed by affine parameter constraints, preserving
convexity, but it changes the model and must be tested in a separate run.
The poor depth/parity/branch outcomes additionally motivate richer sequence
features; adding parameters alone is not an identification argument.

The post-run structural check is reproducible, for example:

```sh
jq -e -f experiments/convex_stress/check_recall.jq \
  experiments/convex_stress/runs/stress-v1/frank_wolfe/basis-recall-easy/seed0/channel.json
```

Validation includes 11 native tests, formatting, clippy with warnings
denied, every frozen dataset hash, all 210 saved-outcome replays, and all
six recall structure checks. Runs used four native CPU workers on a shared
56-thread Xeon E5-2680 v4 host with Rust 1.99.1. No transformer was retrained
in this cohort, and its different optimizer/observation schedule precludes
a matched training-time comparison.
