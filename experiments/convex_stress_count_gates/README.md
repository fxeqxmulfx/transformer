# Shared context mixtures from fixed class-count profiles

Can stronger sharing repair CM46's validation-CE regression? The reviewed local
model kept all35 successes and improved full/minibatch raw counts, but worsened
CE in every mode. This component keeps the CM45 anchor, radius one half, 91-map
bank, complete complex head and ordinary diagnostic CE. It changes only the
fixed gates: five shared rows replace one row per source block. It ports the
reviewed native implementation `a8deaa63` without changing that frozen source.

CM46 at `c60816fa` already proves the required statements for an arbitrary
fixed finite gate family. Here `G=Fin5` and each input has one nonzero gate of
mass one. Thus `weights_mem`, `complex_features_bound`, `ce_convex`,
`ce_quadratic`, and the original-batch identities apply directly. Gates are
computed before any coefficient optimization. No new Lean theorem or unknown
raw-label correctness premise is introduced. The source spectrum supplies
the bound; observed class-frequency profiles are additional model structure,
distinct from the manuscript's spectral balancedness conditions.

For a source block, count labels across its unique observed states. Remove
the query's own state if it is observed. Let `n` be the remaining count and
`m=n-max_label_count`; this definition also handles more than two labels.

| Fixed gate | Remaining count profile |
| --- | --- |
| 0 | Empty or pure (`m=0`) |
| 1 | `0<m/n<=1/8` |
| 2 | `1/8<m/n<=1/4` |
| 3 | `1/4<m/n<=3/8` |
| 4 | `m/n>3/8` |

Integer comparisons implement the boundaries. Numeric raw-label names never
enter the choice. An unseen query keeps all observed counts; a seen query
removes its own unique source state, even when several original records repeat
that input. Discovery still used all training records, so count/context
exclusion is not a refit of discovery. The diagnostic retains every original
record and its full-vocabulary CE denominator.

The coarse gate indexes a local simplex row. The original source block still
indexes the complex feature bank and complete output coefficients. These two
indices must not be confused. Different source blocks may share a local row;
their ordinary CE slopes add with their original record weights. All observed
and unchanged-anchor prediction paths return the exact CM45 parent.

The optimizer is unchanged: anchored per-gate vertex initialization, then
64 full-batch steps or eight original minibatch epochs with actual short tails.
Forward-only stops after initialization and uses no gradients or optimizer
updates. Absent gates remain unchanged. The fixed line bank and improvement
tolerances are in `protocol.json`. Only the lowest complete training diagnostic
at initialization or epoch endpoints is retained; held-out metrics never select
a model. Every profile, input gate, local coefficient, optimizer trace, actual
feature row and work forecast is reconstructed under an external parent digest.

Ten tests pass. They include dense ordinary CE and finite-difference slopes,
original source training in every mode, repeated-record/tail derivatives,
actual complex products and budgets, two genuine source blocks sharing one
gate, absent-gate updates, exact count boundaries, label-renaming invariance,
query-state exclusion despite repeated records, and corrupted/rehashed state.
A boundary fixture has seven majority and one minority source state: removing
the minority gives gate0, removing a majority gives gate2, and an unseen query
keeps gate1. Repeating the minority record does not change its excluded profile.

The inherited 47-record noisy fixture uses one active shared profile and keeps
the earlier CE/raw tradeoff: full CE0.296678to0.293904 with41correct; mini and
forward CE0.358473to0.325070 with42to41correct. This fixture checks the component;
it does not establish an original-suite improvement. The complete comparison
follows exact-commit component review and must retain the negative CM46 result.

Preallocation forecasts include inherited CM45 costs, all label histograms,
per-input gate construction, five local rows, candidates/traces, per-record
diagnostic caches and actual mixed feature rows. Limits remain56GiB forecast,
32billion scalar-work units and4096gradient batches; each process uses the
64GiB address cap. These are conservative executable guards, not formal timing
claims. Generated outputs remain ignored on storage.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_count_gates/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_count_gates/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_count_gates/Cargo.toml --all-targets -- -D warnings
```
