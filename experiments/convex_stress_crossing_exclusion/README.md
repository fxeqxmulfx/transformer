# Exact crossing certificates for exterior raw labels

Can the full hull-intersection criterion identify answers missed by the
observed-point test? This native component implements CM50 at `2b17dc86` and
wraps the reviewed CM49 component at `9455f516`. Original ordinal input-token
coordinates and observed raw labels determine the fixed contexts. No held-out
answer, task name or arithmetic answer formula enters construction.

For each still-uncovered matched query in a separated source block, augment
each candidate class hull with the query. Test other-class hull vertices,
query containment and all pairs of nonparallel hull edges. Exact segment
parameters yield two rational mixtures at the same coordinate. Collinear
contacts are covered by containment. A separate verifier checks original
source identities, distinct raw labels, positive masses, unit totals, the
strictly positive tagged query role only on the candidate side, and both
coordinate identities. All integer arithmetic is checked; overflow refuses
construction. No floating-point tolerance can grant an exclusion.

The candidate pool contains every raw label observed in the block. Exactly
one survivor selects a uniform positive mixture of all its original source
features. The actual mixed complex features multiply every unchanged real
and imaginary coefficient of the complete head. Prediction weights are
independent of the geometric certificate. Confidence remains one and the loss
is ordinary full-vocabulary raw cross-entropy.

All earlier same-label hull rows and observed-point exclusion rows remain
exact, including their saved certificates. Every unsupported, observed,
ambiguous, inconsistent or otherwise inactive query retains its complete
parent prediction. Loading reconstructs every parent and every new decision,
then checks external bindings, pools, both mixtures, prediction rows and costs.

| Mode | Original endpoint | New rule |
| --- | --- | --- |
| Full batch | Retain actual complete trained head and original records | Fixed geometric contexts; zero new gradients or updates |
| Minibatch | Retain order, repeats, every update and short tail | Same construction; zero new gradients or updates |
| Forward only | Retain the original zero-update endpoint | Same construction; no gradients or updates |

Identification remains conditional on convex target regions and membership
of the missing answer in the observed-label pool. Neither premise is proved
by this component or follows from flatness. The protocol explicitly records
both as unproved. Unobserved vocabulary classes can remain additional
survivors, and closed pools can still have ambiguous gaps. Geometry depends
on ordinal token IDs; arbitrary token renaming need not preserve it.

Thirteen native tests pass, including CM50's strict planar crossing, old point/segment
certificates, polygons, triangle containment, ambiguous gaps, missing classes,
inconsistent observations, corrupted certificates and overflow. An oblique
finite-grid comparison uses the independently reviewed Boolean hull-overlap
oracle. Actual full/minibatch/forward training histories retain 45 records
with repeats and a five-record tail. Independent complete source products
agree with feature mixing; ordinary CE and source/head bounds are checked.
Observed batch loss and full-head gradients remain exactly identical.

The resource forecast retains every parent cost and adds original separation,
all containment and edge-pair checks, both certificate verifications, rows,
source storage and serialization. Across all candidate classes these searches
are quadratic in the block's observed size. The construction allowance is
`2048 * sum_eligible (block_size + 1)^2 + 512 * cells * max_states` scalar units.
The buffer reserve adds `4096 * (cells + sum_eligible (block_size + 1) + max_states)`
bytes. The limits remain 256 billion scalar units, 56 GiB forecast buffers and
64 GiB process address space. These native guards are not allocator proofs.
Complete predictions and inherited training costs remain separately charged.

The training-only inspection of all six reviewed task310 source roles succeeds
under the unchanged limits. Counts are identical across the actual three modes:

| Role | Newly identified domain inputs | Still ambiguous | Inconsistent |
| --- | ---: | ---: | ---: |
| Prior | 6350 | 7762 | 0 |
| Supplement | 6540 | 7999 | 0 |

The largest buffer forecast is 9,780,045,312 bytes (9.11 GiB); the largest total
work forecast is 178,756,968,960 scalar units. Actual inspection peak RSS is
3,087,132 KiB (2.94 GiB), elapsed time 71.17 seconds. These are complete-domain
construction counts and costs, not accuracy measurements. Generated evidence
is under `crossing-exclusion-component/coverage-v1` on the storage runs disk.

The coverage executable reads frozen model envelopes only. It constructs the
full input domain without loading evaluation examples or scoring held-out
answers. Original accuracy and CE remain the reviewed task310 results until
the separate adapter and all35 comparison are reviewed and executed. That
comparison must retain both policies, all modes, previous references, the
task298 lower-CE tradeoff, failures, conventional and recorded DiffGPT baselines.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_crossing_exclusion/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo run --release --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion/Cargo.toml --example coverage -- SOURCE_MODEL.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

Generated outputs stay ignored on storage. Component fixtures and domain
coverage are not original benchmark accuracy measurements.
