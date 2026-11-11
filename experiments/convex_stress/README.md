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
so all logits remain linear in the trained joint coefficients. The actual
categorical feature count is `keys * values`; the flat-spectrum theorem
bounds representation through that count, not benchmark accuracy.

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

Scientific runs have not yet started. Preflight regenerated and verified all
35 frozen task datasets. No benchmark result is claimed by the unit tests.
