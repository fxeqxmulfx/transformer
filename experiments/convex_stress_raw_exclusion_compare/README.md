# Convex exclusion on the complete original stress suite

Does the reviewed CM49 exclusion context improve raw-label accuracy and
ordinary cross-entropy beyond task305? This adapter wraps component `9455f516`
and is ported from reviewed task305 at `5dd63c6e`. The complete CM48 prior and
supplement states are its source. All task305 outcomes, task301 guarded CE
results, task296 source comparisons, task298 lower-CE tradeoff, earlier costs
and the failed task301 work-cap attempt remain in the summary.

| Quantity | Protocol |
| --- | --- |
| Original scope | All 35 seed1 tasks and both 105-choice policies |
| Training modes | Original full batch, minibatch and forward only |
| Records | Exact original order, repeats, short tails and raw labels |
| Baselines | All 35 conventional transformer endpoints and 336 depth outcomes |
| New model | Exact augmented-hull conflicts and a complete observed-label candidate pool |
| Eligibility | Current CM48 representation and actual selected model identity |
| Confidence | Fixed at one before scoring |
| New optimization | Zero gradients and optimizer updates in every mode |
| Original success | Unchanged primary `sequence_accuracy >= 0.99` |
| Raw outcomes | Complete primary raw totals, per-task raw counts and ordinary CE |

Only a unique surviving candidate changes an uncovered query's feature row.
Every other parent row is retained, including original observed and covered
inputs. Positive same-label source features are mixed before multiplication
by all unchanged complete real/imaginary head coefficients. Construction takes
training observations and frozen source states; neither held-out labels nor
task names select candidate pools, certificates, confidence or fallback rules.

CM49 requires convex target regions and membership of the unknown query label
in the declared candidate pool. Both remain explicitly unproved in the protocol
and outputs. Original raw accuracy measures this geometric bias; it does not
prove the assumptions. Flatness supplies bounds. Earlier suite results guided
the model's development, so this is not a fresh architecture-selection holdout.

All endpoints are constructed and sealed on disk before new metrics are
computed. Scoring reconstructs each complete state and verifies its saved hash.
The old metrics must reproduce exactly, and the new observed metric object
must equal the original object exactly. Independent replay reconstructs and
loads every saved pool, certificate, prediction row and full parent state.

Each fresh split also audits raw prediction changes, fixes, regressions and
errors among newly identified queries. Its independent raw correct counts must
match the original metric implementation. Those additional complete predictions
are charged. Full-suite raw totals use native raw counts where pooled-field
metrics coexist; all 35 primary denominators remain present in every mode.
Every one of the 210 choices is compared against both the complete task305
and task301 references. Earlier low-CE/high-error outcomes remain visible.

Only one endpoint is live at a time. Its prior and supplement forecasts are
summed; the complete case reserves three constructions per endpoint. Limits
are 56 GiB forecast, 64 GiB process address space, 256 billion scalar work per
component, 3 trillion per case and 64 billion for metrics plus raw audits per
endpoint. At least 16 GiB of free storage is required. Failed or refused cases
remain in the denominator and prevent a complete success summary.

Tests cover both policies, all modes, complete reference/cost preservation,
independence from held-out labels and scores, original source/record identities,
actual nonzero complex products, changed-certificate rejection, disk sealing,
raw-count conventions and explicit regression reporting. The component's
separate 12 tests establish exact geometry and original full-head gradients.
Fixture passes are not original benchmark gains.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_raw_exclusion_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion_compare/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_exclusion_compare/Cargo.toml
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
bash experiments/convex_stress_raw_exclusion_compare/run_original.sh FROZEN_EXECUTABLE REQUEST.json FRESH_OUTPUT
jq -f experiments/convex_stress_raw_exclusion_compare/report.jq FRESH_OUTPUT/summary.json
```

The request supplies the producer commit; absolute task305 source directory,
completed manifest and exact review files; task308 component review files;
task301 reference root, manifest and review files; and the failed attempt's
root and manifest. The protocol pins every identity. Freeze the source,
executable and request before original scoring. Generated states, metrics and
logs remain ignored on storage; authored findings follow in `original/README.md`.
