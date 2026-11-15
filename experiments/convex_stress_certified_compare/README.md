# Full comparison with certified complete-head evaluation

Does the separately proved and tested CM63 evaluator remove task347's numerical
refusals without losing accuracy elsewhere? Repeat the same13 original cases,
training data and six endpoints. This producer changes numerical evaluation only.
It does not add training labels, choose a target program, alter discovery, modify
the feasible set, or replace ordinary cross-entropy.

| Item | Protocol |
| --- | --- |
| Model component | `convex_stress_certified_head` atd1b10e2e, after18 CM63 Lean proofs at18308305 |
| Discovery and training | Unchanged task347 implementation at1371ae4f. Reconstructed complete training state must hash to the original347 seal after restoring its old numerical forecast. |
| Endpoints | Coordinates and paired Categorical control, each with actual Forward, Full and Mini policies |
| Original data | All13 cases and every original train, validation, test, longer-length and novel-label split; no filter of difficult examples |
| Decisions | Preserve original95% sequence success; separately report99% raw token accuracy,99% whole-sequence accuracy and their conjunction on every heldout split |
| Numerical decisions | Outward numerator enclosures for separated winners; independent exact rational real/complex contractions for ambiguity; first exactly maximizing class on genuine ties |
| Evidence | Complete saved parameters, all class logits, ordinary teacher-forced CE, own-output generation, numerical certificates, original-row recount, replay and independent rebuild/repeat |

The previous77/165 coverage is a union of separate frozen seed1 cohorts. It is
not a shared165-task checkpoint, three seeds, or a matched-budget transformer
comparison. Original transformer results remain in the unchanged inherited
reports. This comparison retains complete344/346/347 reports, their manifests
and verification, the original35 report and the remaining historical inputs.
Refused endpoints remain in the denominator and receive no score credit.

The full runner recomputes all saved predictions, certificates, CE and integer
counts from raw outputs. Its tests reject altered logits, maximizer lists and
scores even when a modified file's saved hash is replaced. Separate fixtures
cover wrong early EOS, extra generated tokens, unseen labels, ignored BOS targets,
transition counts, incomplete manifests, genuine ties and the minibatch tail.

Each process runs under64GiB; forecasts retain56GiB buffer admission. Besides the
existing interpreter/scalar budget, each endpoint admits4096 completed exact
fallback queries. The first excess query is explicitly refused and included in
the worst-case4097-query work reserve. Exact integer precision is capped at8192
bits; coefficient multiplication counts are reported separately from their
nonconstant big-integer limb cost. Data generation, complete search, training,
raw serialization, certificate recomputation, replay and hashing contribute to
measured time/RSS. No fixed CPU-time bound is inferred from these counts.

The baseline347 verification now independently reproduces all218 artifacts.
Its actual SHA256, `4f40b9f20d216f5d1564ff76683733adfd5e3c08d681b0f44b116c82b95ff002`,
is pinned in `protocol.json` before this producer is frozen. All12 native checks,
strict Clippy and the release build pass. The full comparison and independent
repeat follow this freeze; no349 heldout accuracy or additional catalog pass is
claimed until they complete.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/certified-compare-v1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_certified_compare/Cargo.toml -j2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_certified_compare/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_certified_compare/Cargo.toml -j2
# After the complete protocol is committed and347 has independently verified:
bash experiments/convex_stress_certified_compare/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-certified-compare"
bash experiments/convex_stress_certified_compare/run.sh FROZEN_BINARY FRESH_ORIGINAL original PRODUCER_COMMIT STREAM347_ROOT VERIFIED347_JSON
bash experiments/convex_stress_certified_compare/run.sh REBUILT_BINARY FRESH_REPEAT repeat ORIGINAL_ROOT
prlimit --as=68719476736:68719476736 -- REBUILT_BINARY verify ORIGINAL_ROOT REPEAT_ROOT FRESH_VERIFIED_JSON
```

Both run directories must be fresh. All13 training states and all six endpoints
are sealed before any scoring. Rebuild from the same source checkout path into a
fresh Cargo target directory for executable byte comparison: Rust binaries can
embed checkout-dependent source locations. Keep both build logs and hashes.
Generated evidence belongs in persistent ignored storage, not in the worktree.
