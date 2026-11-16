# Complete compact comparison

Can the CM70 compact model receive the entire fixed composition family and
every original record under the 110 GB RAM limit, including cases refused by
the former dense-model estimate? This separate prospective comparison uses
the Lean CM64–70 model and the input family from task358. It preserves the
existing bounded, reduced and composed comparison producers.

Following the user's 2026-11-16 priority, new comparisons require minibatch
and forward-only with minibatch processing. Full batch is an optional
reference, and existing full-batch baselines stay intact. The model's native
regression controls continue to cover all three numerical policies.

The preparation component is implemented. It independently repeats the full
candidate partition, constructs every packed observation, rebuilds the complete
observed dictionary and rehashes the actual stored states and labels. All
962,806 candidates remain in a compatible constant control, including equal
functions as separate candidates. No original record, class or unknown row is
removed. Ordinary CE and the numerical training policies belong to the compact
model; preparation supplies no unseen labels or task identity.

| Pass | Admission and retained evidence |
| --- | --- |
| Candidate verification | Original fixed family and independent interpreter; complete partition and original audit must equal the frozen probe |
| Materialization | Original program-major callback order, checked UInt16 words, every original ID, full dictionary indices and actual-storage hashes |
| Work | Each pass retains its own 40-trillion ceiling; report both counters and their complete sum, including on refusal |
| Memory | Count actual training, selection, family and label-vector capacities; reserve bounded generator/serialization/interpreter scratch before allocating; subtract inputs before future model construction |

The original p193 constant probe charges 38,699,598,835,200 operations for
candidate verification alone. Reusing the previous single 40-trillion
preparation allowance would leave too little work for complete materialization.
The separate passes retain every old charge and a bounded total of 80 trillion.
The older frozen producers and their counters remain unchanged.

The prospective endpoint work ceiling is 256 trillion, separately from
preparation. For the largest original constant input, the existing 127-record
minibatch policy needs 146 updates. Their mandatory charges alone total
54,949,599,919,456, before scalar solves, contractions, initialization, audits
and checkpointing. Thus the former 40-trillion allowance cannot complete this
policy. The new limit changes admission only: every charge, update and short
tail remains, and releasing a checkpoint reserve never resets used work. It
is a bounded allowance, not a claim that every model will fit or finish.
Native controls compare the lower bound with actual training counters and
check complete tail updates and hard-limit rejection.

Native controls compare the entire materialization with the prior encoder:
every state and raw label, exact dictionary fields and the sum of both work
counters. A four-record control includes duplicates and a singleton interval.
It retains the complete 962,806-candidate family. Tight memory admission stops
before candidate work, a missing candidate is rejected, and malformed or
over-budget phase counters fail. Fixed scratch reservations are checked against
the pinned grammar size and bounded interpreter tables.

Checkpoint and input-only prediction boundaries also use compact admission.
Reading reserves the additional stored dictionary until comparison, retained
parameters during head construction, and I/O/audit scratch. Prediction reserves
the full family generator, selected program capacities, query states, bounded
exact-integer and interval scratch, and two raw answers during replay. State
construction moves training metadata rather than cloning it. The caller must
subtract all original inputs and runner metadata before these nested limits.

Sixteen native controls cover complete preparation, work limits, real checkpoint
files, two required mode forecasts, and both-disk refusal. Complete old and new
models agree on parameters, CE, actual update traces, every queried answer and
work charge. The 130-record control keeps the 127-record minibatch and three-row
tail. Raw readback preserves 99/100 and 98/100 distinctions, all wrong-label and
numerical-refusal denominators, and rejects rehashed logit corruption and oversized
lines before prediction. These are synthetic controls, not original-task results.

The complete runner is implemented. It retains all 54 case entries and requires
both training endpoints for each case before the global heldout barrier. Typed
probe metadata and the exact original input hashes are checked; every legacy
probe refusal remains visible beside the compact input forecast. The forecast
admits inputs only. Each training, save, read and prediction phase still needs
its full local capacity check. Preparation counters are preserved in training,
scoring and replay, and scoring must reconstruct the same complete inputs.

The caller ledger counts actual inventory, dataset and endpoint capacities,
with conservative tree-node, parser and serialization reservations. Before a
new large run, existing experiment pipelines must finish; their wrappers are
checked as well as their current child processes. Other process RSS and a 2 GiB
growth reserve are subtracted from the 110 GB allowance. Runtime limits and
measured process peaks go in local logs. These checks are native resource
controls, not a Lean theorem about RSS.

A synthetic runner control uses 130 distinct training inputs and 159 heldout
inputs. Training succeeds while the heldout file contains invalid JSON, scoring
fails before the global barrier, and the barrier rejects a missing case among
54 synthetic entries. After restoring the bound dataset and sealing all 108
required endpoints, the runner scores and replays every row. It checks omitted
modes, erased preparation charges, changed metadata and the two-mode report
fields. Its other 53 entries are explicit synthetic refusals; this control does
not stand in for the original 48 available datasets.

Task358's complete independent verification is now pinned in the protocol:
all 102 artifacts match and all 48 available training supports are nonempty.
Production is enabled only for these exact inputs; commit and freeze the checked
producer before original fitting. No original fitting or new accuracy claim
exists yet. The disk forecast reserves every full logical
checkpoint and raw prediction for both runs. A shortage of either disk stops
execution. New coverage fields explicitly refer to the two required modes;
historical three-mode coverage stays separately identified.

The typed probe reader also retains complete reports above the former 64 MiB
limit while enforcing the pinned 128 MiB cap. It rejects changed work counters,
metadata, hashes, unknown fields, and a complete-length but forged partition.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/compact-compare-v1
cargo test --offline --locked --manifest-path experiments/convex_stress_compact_compare/Cargo.toml -j2
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_compact_compare/Cargo.toml -j2 -- -D warnings
cargo test --offline --locked --release --manifest-path experiments/convex_stress_compact_compare/Cargo.toml -j2
bash -n experiments/convex_stress_compact_compare/run.sh experiments/convex_stress_compact_compare/freeze.sh
```

After pinning the verified probe, rerun the checks, commit, build the release
binary from that clean commit and create a fresh freeze before original fitting:

```sh
cargo build --offline --locked --release --manifest-path experiments/convex_stress_compact_compare/Cargo.toml -j2
bash experiments/convex_stress_compact_compare/freeze.sh NEW_FREEZE RELEASE_BINARY
bash NEW_FREEZE/source/experiments/convex_stress_compact_compare/run.sh \
  NEW_FREEZE/convex-stress-compact-compare WORKSPACE NEW_ORIGINAL original \
  ORIGINAL_DATA VERIFIED_COMPOSITION_PROBE RETAINED_SUMMARIES PRODUCER_COMMIT
bash NEW_FREEZE/source/experiments/convex_stress_compact_compare/run.sh \
  INDEPENDENTLY_REBUILT_BINARY WORKSPACE NEW_REPEAT repeat NEW_ORIGINAL
INDEPENDENTLY_REBUILT_BINARY verify WORKSPACE NEW_ORIGINAL NEW_REPEAT NEW_VERIFICATION_JSON
```

The independent rebuild uses the frozen sources and a separate Cargo target
directory. Before new coverage is credited, compare the complete deterministic
input, checkpoint, trace, raw prediction and summary manifests. Keep generated
outputs outside Git and preserve the existing bounded/reduced/composed runs.
