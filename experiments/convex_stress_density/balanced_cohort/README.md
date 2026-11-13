# Balanced original-task density follow-up

This separately frozen CM24 follow-up asks whether observable feature balancing
and explicit missing-axis completion improve the original nine categorical
pair tasks. Its harness is ported from `../cohort` at `3311e13c`. The original
dataset generator, split fingerprints, metrics and full-vocabulary CE are
unchanged. The constructor receives training cell/label pairs only.

| Candidate family | Policy |
| --- | --- |
| Complete CM21–23 parent | Rebuild every earlier CM15 trial, exact and interval attempt, and certified refinement with the original policies. |
| Balanced uniform density | Normalize each nonzero supplied feature row to squared norm equal to rank; audit the explicit uniform density. Its kernel column is an empirical proposal with no objective-optimality claim. |
| Balanced interval densities | Tolerances `0,0.1,0.3`, 4,096 iterations each, 73,728 total; first two observed rows are the objective anchors. Keep all accepted, refused and capped outcomes. |
| Missing phase entries | Sort only observed nonzero distinct phases; assign unused ranks to missing axis entries in index order. This arbitrary completion supplies no unseen labels. |
| Certified refinement | At most 24 rounds per seed, 16 seeds, 256 assignments; preserve all integer duals, decoder counts and states. At most 24 new seed audits. |

Lean `75ca9276` proves actual normalization, uniform-density feasibility,
unchanged observed CE under unseen-axis completion, and unchanged latent
spectral budgets. It also proves a capacity limit: a surjective phase axis
cannot represent a constant opposite row alongside nonconstant rows. This
follow-up keeps that narrower native family and tests the numerical repair;
it does not claim to solve the capacity limitation.

Near-zero means measured row energy at most `1e-24` in the native policy.
Such a row is refused when observed, and explicitly replaced with ones when
unobserved. This numerical convention is recorded separately from the exact
Lean zero-row statement. No re-centering or re-orthogonalization follows
balancing. Every accepted matrix and candidate has raw numerical checks;
none is described as an exact real-arithmetic certificate.

All parent candidates precede new candidates on CE ties. Selection is sealed
before split metrics. Full original-input replay reconstructs every feature,
missing-axis mask, matrix, assignment certificate, decoder and cost ledger;
raw all-frequency DFT/SVD/CE and original split metrics are recomputed.
The harness retains every process failure and all 27 references to the nine
forward constructions. These references are not three optimizer runs.
Full35/modes/transformer integration waits for complete verified parents.

Checks and reproduction use native tooling only:

```sh
cd experiments/convex_stress_density/balanced_cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
cd ../../..
BALANCED_PRODUCER_COMMIT=<committed-producer> \
BALANCED_BINARY=<frozen-binary> \
  bash experiments/convex_stress_density/balanced_cohort/run_cohort.sh \
  experiments/convex_stress_density/runs/balanced-original-v1
```

Freeze the tested committed source, full source fingerprint, binary hash,
protocol and launcher before preparing original data. The launcher is serial,
uses one BLAS thread, enforces the 64 GiB process limit, reserves 16 GiB disk
space and refuses existing destinations. Partial artifacts remain local.
Generated evidence stays ignored in the root experiment's runs directory.
