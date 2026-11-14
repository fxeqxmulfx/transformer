# Anchored local context mixtures

Can input-dependent local context preferences reduce the remaining raw-label
errors while keeping ordinary CE convex in the learned coefficients? CM46,
proved and reviewed at `c60816fa`, supplies the actual product identities,
complex feature and head bounds, convex local CE, explicit slopes, and exact
original-batch identities. This Rust component implements those local rows.
The original 35-task comparison is a separate step after component review.

The parent is the reviewed CM45 component at `14b8ad6e`. Its 91-map bank,
shared mixture, complete complex head, observed records, and mode metadata
are fixed. Each input's existing latent-block identity is its fixed one-hot
gate. The effective bank weights are
`0.5 * parent_weights + 0.5 * local_weights[block]`. Every local row is a
simplex. No additional gate discovery, output decoder, input arithmetic rule,
or jointly trained head is introduced. The source spectrum gives the feature
bound; the inherited ordinal context rules remain an additional model bias.

| Mode | Initialization | Subsequent local optimization |
| --- | --- | --- |
| Forward | For each gate, keep the anchor row unless an anchored bank vertex improves diagnostic CE | None; zero derivatives and optimizer updates |
| Full | Same local initialization | 64 conditional-gradient steps on all original records |
| Mini | Same local initialization | Eight original-order epochs at the parent's minibatch size, with actual short tails |

The source-excluded diagnostic is the same as CM45: every original raw record
is evaluated against a fixed canonical margin-24 conjugate head. Sparse active
class storage retains every absent class's exponential in the full vocabulary
denominator. All training records participated in latent discovery; excluding
the current source state is not a leave-one-out refit. Optimization transfers
these local coefficients to the unchanged actual complete head. It does not
prove an unseen raw target is correct.

Initialization scores the anchor and every anchored bank vertex separately
within each gate, retaining original multiplicities. Empty gates keep their
anchor. Iterative steps minimize the explicit radius-scaled local slopes and
search the fixed line bank `[0,1/16,1/8,1/4,1/2,1]`. Gates absent from a
minibatch are unchanged. The saved endpoint is the lowest complete diagnostic
CE among initialization and epoch endpoints. Vertex/epoch improvements require
more than `1e-10`; line improvements require more than `1e-12`. Ties retain
the earlier candidate. Raw accuracy is reported separately from this CE rule.

Actual inference mixes the source's complex Fourier features before applying
every original real/imaginary coefficient. Original observed rows use the exact
parent path, preserving losses and complete-head gradients on repeated records
and tails. Unchanged local rows also return the parent exactly. All fixed gates,
anchor, radius, local coefficients, selection traces, feature rows and costs
are reconstructed when loading, under an external digest of the complete CM45
parent. Rehashing a corrupted parent does not bypass its own reconstruction.

The eight native tests check full/minibatch/forward work counts, dense ordinary
CE and finite-difference slopes, absent gates, opposing gate preferences, actual
nonzero complex products and budgets, two genuine latent blocks, original
trained-head derivatives and repeated/tail batches, saved-state corruptions,
and unchanged or invalid domains. The 47-record noisy fixture gives:

| Mode | Parent diagnostic CE | Local diagnostic CE | Parent / local correct raw records |
| --- | ---: | ---: | ---: |
| Full | 0.29667798 | 0.29390434 | 41 / 41 |
| Mini | 0.35847252 | 0.32506951 | 42 / 41 |
| Forward | 0.35847252 | 0.32506951 | 42 / 41 |

These are component fixtures, not original stress-suite scores. They retain
the CE/raw-label tradeoff rather than treating lower CE as a raw-accuracy
guarantee. Full makes 13 local-row updates; mini makes 48 and retains its
initial local selection; forward makes no gradient or optimizer calls.

`forecast` checks a 56 GiB buffer limit and 32 billion scalar-work guard before
new diagnostic/row allocation, with a maximum of 4096 gradient batches. It
includes the inherited CM45 bounds, per-record bank logits, all local rows and
candidates, per-step gate choices, fixed gates, and actual mixed feature rows.
Construction counts sorting and context weights; optimization counts anchor
and vertex scoring, complete local gradients, line scoring, epoch scoring,
and local-row update/digest work. These conservative native guards are not
formal runtime bounds or equal-compute comparisons. Runs use a 64 GiB address
limit. Generated evidence stays ignored on the storage disk.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_gated_context/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_gated_context/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_gated_context/Cargo.toml --all-targets -- -D warnings
```

After exact-commit component review, the original comparison must freeze all
35 tasks, both policies and all three modes before scoring. It must reproduce
the CM45 endpoints, retain the conventional transformer and depth baselines,
seal local coefficients before held-out evaluation, and report every raw-label,
CE and original-success regression. Existing frozen experiments stay unchanged.
