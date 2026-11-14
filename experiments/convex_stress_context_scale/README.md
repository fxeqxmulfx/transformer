# Fixed confidence on complex context features

Can training-only confidence calibration reduce ordinary raw-label CE while
retaining the reviewed CM42 context predictions? CM43 at `2dcaa38a` proves
the actual scaled-logit identity, feature/head bounds, raw-order equivalence,
convex ordinary CE at fixed scales and exact original observed batches.
This native component wraps the frozen CM42 implementation at `c62022c5`.

| Component | Policy |
| --- | --- |
| Neighborhoods | Retain the complete reviewed CM42 context map |
| Confidence candidates | 1, 0.75, 0.5, 0.375, 0.25, 0.125, in that order |
| Selection | Lowest canonical training-record mean CE; improvement greater than 1e-10 |
| Canonical logits | Original margin24 weighted raw-label votes, independent of trained coefficients |
| Query exclusion | Remove the queried observed state; preserve original record multiplicities |
| Actual inference | Scale the actual complex features before the complete original coefficient product |
| Observed and unmodified rows | Exact original logits and coefficient derivatives |
| Updates | None beyond the inherited full/minibatch/forward endpoint |

The scale-one diagnostic must reproduce the reviewed context selector's
corresponding candidate. Latent discovery and neighborhood selection still
use all original training data; this diagnostic does not refit discovery in
each leave-one-out split. The ordinal context metric remains an explicit
locality assumption. Flatness bounds the source features and the unchanged
full coefficient budget; it does not identify unseen labels.

The chosen positive scale is used only on previously interpolated unseen
ambiguous matched cells. Other inputs retain scale one. All records, source
coefficients, context weights, scale candidates and allocation/work forecasts
are bound to the externally supplied context-model digest. Loading rebuilds
the context and selection before accepting serialized state. Native limits
are checked forecasts, not formally verified floating-point error bounds.

Positive scaling preserves exact-real raw ranks for this fixed context map.
Floating-point predictions and many-to-one pooled task metrics require fresh
measurement. Shrinking can increase CE on correct predictions; training-only
selection supplies no held-out improvement guarantee. No original-data result
is asserted by this component. The separate original comparison must keep
all35 tasks, both policies, all three modes and the retained baselines.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context_scale/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context_scale/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context_scale/Cargo.toml --all-targets -- -D warnings
```

Fixtures check the actual scaled complex products for every candidate, raw
argmax and feature bounds, the CE mixture inequality, coefficient-independent
selection, repeated/tail batch identities in all three actual training modes,
external-binding mutations and unchanged unsupported/prediction domains.
