# Saved-parent fallback with a restricted input domain

Can a frozen predecessor help a causal candidate without limiting the candidate
to that predecessor's shorter context? This implements CM41, proved in
`ConvexDepth/PartialFallback/{Basic,Batch}` at `baf7c2e4`. The original CM38
component at `027237cd` supplies checked complex heads, original support,
external tensor bindings and combined budgets. Its source remains frozen.

The initial task 277 comparison exposed a domain error before scoring some
original long splits: a saved count parent has a finite value alphabet tied to
its saved context. The candidate's actual causal prediction function accepts
prefixes through 4096. This component changes the fixed routing rule as follows.

| Candidate support | Parent available | Actual logits |
| --- | --- | --- |
| Supported | Either | Candidate |
| Unsupported | Yes | Saved parent |
| Unsupported | No | Candidate |

The parent domain comes solely from its checked saved representation. Counts
retain their saved context; Causal parents retain their original 4096-prefix
prediction domain. Every call validates the nonempty prefix, full vocabulary
and length limit before computing support. Route diagnostics keep candidate
support, parent availability and the actual source separate. An unavailable
parent is never evaluated, and its finite alphabet is never expanded.

The model stores the exact previously checked heads. It adds no coefficient,
feature discovery or optimizer update. Both blocks and original record buffers
retain the original charged bounds. The core constructor still requires the
saved parent to cover the declared training context; prediction availability
is checked separately for each input. Loading binds the exact routing version,
original records, parameters, budgets and external source digests. Unknown
fields and changed embedded bindings are refused.

CM41 realizes this rule with actual disjoint feature coordinates and a zero
extension of the partial parent. Flat complex component codes bound those
coordinates. The combined coefficient budget includes both stored heads, and
ordinary raw-label cross-entropy is convex with the parent block fixed.
Candidate-supported original records have exactly the same normalized loss
and candidate-coordinate derivatives. Record IDs retain repetitions and short
minibatch tails. Forward-only construction keeps the original zero-update state.
Neither support nor availability certifies the answer at an unseen input.

Native tests cover all three routes with distinct actual complex predictions,
Counts and Causal parents, boundaries at 5/6 and 65/66, and prefixes through
4096. The old short-parent refusal is reproduced explicitly. Tests also check
saved-state mutations, both-head budgets, input and record caps, all real and
imaginary finite-difference gradient coordinates, repeated records, actual
full/minibatch/forward training histories and unequal batch tails. This is a
component check; it has no new original benchmark score.

Build and check from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_partial_fallback/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

Numeric processes retain the 64 GiB address-space cap and the combined
56 GiB buffer forecast. Original record and token limits remain 200,000 and
20,000,000 respectively. A complete comparison must additionally retain its
uniform metric-work, artifact and disk limits before scoring.

After committing, freeze the clean exact dependency graph and test executable:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/partial-fallback-v1
bash experiments/convex_stress_partial_fallback/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
prlimit --as=68719476736:68719476736 -- \
  "$freeze/partial-fallback-tests-v1" --test-threads=1
```

Keep source archives and test logs local on storage. A new complete comparison
will use this reviewed component with the existing selection rules and separate
support/availability/route masks. The original 277 failures and the frozen
255/261/267 runs remain separate evidence.
