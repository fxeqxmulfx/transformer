# Bounded complex decoder on the complete original pair cohort

Can the CM64 bounded joint decoder improve raw accuracy and ordinary
cross-entropy on more original stress cases while preserving its full convex
domain? This comparison follows the36 Lean results and the reviewed, fixed
4,767-program training-only probe. It measures all54 original arithmetic cases:
48 available datasets, including22 older controls and26 additions; six exact
source helpers remain unavailable. The family keeps every compatible candidate.
Seven additional cases have empty compatible support in the frozen probe.

The model receives only vocabulary and original ordered training triples, then
query pairs. The runner alone sees case names, historical transformer results
and dataset metadata. All full learned masses, decoder coefficients, real head
coefficients and imaginary head coefficients are saved as binary64 values.
The complete head is rebuilt from the saved joint tensor on reload. Predictions
use its actual complex products and independently checked input programs.
Unknown decoder states remain free variables; conditional observed-anchor
coverage is reported separately from measured accuracy and is never labeled
as target realizability.

| Item | Frozen policy |
| --- | --- |
| Family | All4,767 fixed pair programs; exact elimination only for observed label conflicts |
| Confidence | Total shared mass in[1,32], initialized at8 |
| Forward | Zero gradient updates |
| Full | Four complete-record CE updates, learning rate1,000,000 |
| Mini | One epoch of consecutive127-record batches, actual final tail, same rate |
| Data | Original seed1 row order and disjoint half-table splits, pinned by reviewed hashes |
| Scoring | Every original train/validation answer;99% raw and exact, original sequence criterion |
| Admission | 110,000,000,000-byte process limit;48GiB core buffer forecast plus2GiB runner slack |
| Disk | Both filesystems checked; original reserves the full independent repeat,2GiB workspace and16GiB storage slack |
| Work |40T per preparation or endpoint phase, including serialization reserve; exact fallback has a finite query/precision cap |

Every case and all three endpoint outcomes must be sealed before any heldout
pool is parsed for scoring. Refused models remain in the54-case denominator.
Numerical query refusals retain every original target, receive no pass credit
and leave CE explicitly unavailable. Filesystem errors and insufficient disk
space stop the run. Partial evidence is preserved. No compatible candidate,
original record, decoder row or head coefficient is truncated to fit a budget.

Each raw row stores all class logits, original target, maximizers and numerical
certificate. Exact fallback stores a common-maximum-shifted complete logit
vector, preserving CE and ordering. Readback recomputes every prediction and
independently recounts counts and CE, including wrong answers and refusals.
Training traces keep actual update ranges and the short minibatch tail.
Complete manifests bind every input, tensor, raw row, seal and summary. An
independent source rebuild repeats the entire workflow before new catalog
credit is accepted. Executable hashes are retained separately because checkout
paths can change binary bytes without changing deterministic results.

The previous77 verified cases and original35 accuracy/CE summaries remain
explicit historical baselines. Their old raw outputs were deleted at the
user's request after summary preservation. Their verification records describe
those previous reviews. This run freshly verifies its own complete outputs.
It reports union coverage and coverage with the revised22-case control cohort,
including regressions. New constant power-zero controls are separated from new
nonconstant cases. Historical transformer measurements remain references;
there is no new matched-budget transformer training or three-seed claim.

Ten native controls cover bit-exact tensor persistence; modified/nonfinite
coefficients; both-disk failures; complete storage forecasts; actual full,
mini and forward policies; duplicate records and short tails; the exact99%
boundary; target-independent predictions; wrong-label and refusal denominators;
rehashed raw tampering; missing training seals; inconsistent score credit;
and original split integrity. These controls supply no dataset accuracy result.
The allocation-only forecast is30,247,354,063 bytes per run, including all input
copies and metadata slack. Full original plus repeat is60,494,708,126 bytes,
before the declared filesystem reserves. No heldout model score was read to
choose this policy.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/bounded-compare-v1
cargo test --offline --locked --manifest-path experiments/convex_stress_bounded_compare/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_stress_bounded_compare/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_bounded_compare/Cargo.toml
# Commit before freezing or fitting any original model.
bash experiments/convex_stress_bounded_compare/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-bounded-compare"
bash experiments/convex_stress_bounded_compare/run.sh FROZEN_BINARY WORKSPACE FRESH_ORIGINAL original DATA348_ROOT PAIR_PROBE_ROOT RETAINED_SUMMARIES PRODUCER_COMMIT
# Independently rebuild the frozen source into a separate target directory.
bash experiments/convex_stress_bounded_compare/run.sh REBUILT_BINARY WORKSPACE FRESH_REPEAT repeat ORIGINAL_ROOT
prlimit --as=110000000000:110000000000 -- REBUILT_BINARY verify ORIGINAL_ROOT REPEAT_ROOT FRESH_VERIFIED_JSON
```

Flat spectrum supplies the complete-head capacity bound. The finite input-only
family is a separate structural prior. Neither exact-real Lean statements nor
the native projection audits prove unknown-target realizability, floating-point
optimizer convergence or success on an unsupported task.
