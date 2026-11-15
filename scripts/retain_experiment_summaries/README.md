# Retain experiment code and compact findings

The user requested deletion of old experiment outputs on2026-11-15, retaining
code and summaries. This native tool prepares those retained artifacts; it
never deletes an input. The caller must explicitly identify inactive roots,
protect active run dependencies, and verify the archive before removing data.
It uses no Python.

The archive preserves source text and small source configuration files exactly,
deduplicated by SHA256. It retains authored README files, historical verification
records, compiler/producer identifiers, exit codes and resource summaries.
Generated JSON findings are compacted while streaming past tensors and raw
observations. Scalar counts, loss, accuracy, refusal reasons, original success
flags and per-case/per-mode summaries remain. Omitted arrays and objects carry
explicit entry counts. Long curves retain their first and last points rather
than pretending to retain every sample. Original source-file hashes bind each
summary to the former complete report.

`index.jsonl` maps every retained original path to its object hash and mode.
The `summaries/` tree mirrors original paths for readable access to compact
results and historical provenance. `objects/` stores deduplicated contents;
source files can be reconstructed using the index. `archive-summary.json` gives
the completed archive counts and index hash. A successful exit is required
before cleanup; any unreadable input, oversized source file or hash conflict
stops retention. Empty or malformed historical JSON reports retain their exact
hash and parse error with `incomplete_report: true` and no invented score.
The streaming input limit is 2 GiB per JSON report; this includes the existing
1.14 GB signed-comparison report. The separate process RAM limit still applies.
Symlinks are recorded without following them.
Compiler/dependency caches are excluded.

Historical verification records describe checks that completed before cleanup.
After raw outputs are removed, those records are not evidence that the old raw
artifact verifier can still run. Reproduce the experiment from its retained
code and documented inputs if a new raw audit is required.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/summary-retention-v1
cargo test --offline --locked --manifest-path scripts/retain_experiment_summaries/Cargo.toml
cargo clippy --offline --locked --manifest-path scripts/retain_experiment_summaries/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path scripts/retain_experiment_summaries/Cargo.toml
prlimit --as=110000000000:110000000000 -- "$CARGO_TARGET_DIR/release/retain-experiment-summaries" ABSOLUTE_ROOT_LIST FRESH_ARCHIVE
```

Four controls check retention of exact counts and failed outcomes, explicit
large-array omissions, and agreement between the streaming visitor and an
independent complete-tree compactor. Incomplete JSON is explicitly unscored,
while actual filesystem read failures still stop retention.
