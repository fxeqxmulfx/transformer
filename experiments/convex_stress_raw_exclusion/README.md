# Identify exterior raw labels by convex exclusion

Can observations identify more raw answers outside the same-label hulls used
by task305? This Rust component implements CM49 at `26ec2faa`, wrapping the
exact reviewed raw-region model at `0fee1a9d`. It uses the original ordinal
input-token coordinates and observed raw labels inside each routed source
block. No arithmetic answer formula or held-out target enters construction.

For each uncovered matched query in a separated ambiguous block, the candidate
pool is all distinct raw labels observed in that block. For each candidate,
construct its observed hull augmented with the query. A differently labelled
observed point inside this hull rules out the candidate under convex target
regions. Checked integer geometry gives a rational certificate; an independent
verifier checks the original witness identity, its different label, the query's
explicit role, positive masses, unit total and exact coordinate identity.
Testing observed points is sufficient but may miss other hull intersections.

Exactly one survivor enables a new context. Zero or multiple survivors retain
the complete parent, as do observed inputs, existing hull coverage, unknown
source matches and overlapping or inactive blocks. The surviving label chooses
a uniform positive mixture of all its original observed source rows. These
prediction weights are separate from the geometric conflict weights and need
not reproduce the query's input coordinate. Every logit is the actual product
of the mixed complex features and all unchanged real/imaginary head coefficients.
Confidence is fixed at one; ordinary full-vocabulary raw CE is unchanged.

The correctness statement remains conditional. Convex target regions and
membership of the unknown answer in the observed-label pool are explicit
assumptions. The protocol records both as unproved. Adding a previously
unobserved class from the original vocabulary can leave another surviving
answer, as the Lean counterexample and native test demonstrate. A closed pool
alone also leaves some gaps ambiguous. Flatness supplies feature bounds and
does not establish either identification assumption. The input geometry also
uses ordinal token IDs, so arbitrary token renaming need not preserve it.

| Mode | Retained head training | New context |
| --- | --- | --- |
| Full batch | Exact original endpoint and records | Fixed exclusion certificates; zero new gradients or updates |
| Minibatch | Exact order, repeated records and short tail | Same fixed construction; zero new gradients or updates |
| Forward only | Original zero-update endpoint | Same construction; zero gradients and updates |

Twelve native tests check exterior and triangle certificates, ambiguous and
off-line queries, missing classes, inconsistent observations, integer overflow,
certificate mutations and independent positive prediction weights. The nonzero
complex fixture retains 45 original records, one repeated observation and a
five-record tail. Its two exterior queries are identified in all three actual
training modes. Independent source-logit averaging agrees with mixing features
before the complete product; CE, coordinate and coefficient bounds are checked.
Original batch losses and full-head gradients agree exactly. Saved-state tests
reject altered pools, witnesses, source blocks, weights, counts, costs, bindings
and a rehashed invalid parent. These are component checks, not new benchmark
accuracy measurements.

Construction forecasts include all eligible query/candidate scans, augmented
hulls, independent certificate checks, source-point storage and serialization
reserves. Parent costs remain charged. The new work guard is 256 billion scalar
units; the buffer limit remains 56 GiB and each process uses the 64 GiB limit.
Forecasts are conservative native resource guards, not Lean allocator proofs.
Prediction forecasts include the maximum prefix scan and complete source/head
products. Integer-work counters and inherited training costs stay separate.

Loading reconstructs the complete parent and all decisions, checks the external
parent digest and protocol binding, and compares pools, exact witnesses, positive
prediction rows, fallback decisions and costs. The coverage example reads saved
source envelopes only, without creating evaluation data or computing held-out
metrics. A separate frozen comparison must retain all 35 tasks, both policies,
all modes, the task305/301/296 references, task298 CE tradeoff, all failures and
both conventional and recorded DiffGPT baselines.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_raw_exclusion/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo run --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion/Cargo.toml --example coverage -- SOURCE_MODEL.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

Generated models, reports, timings and logs stay ignored on storage. Original
raw accuracy and CE remain the reviewed task305 values until the separate
complete comparison evaluates this component.
